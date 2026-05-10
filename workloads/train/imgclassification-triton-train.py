"""
ResNet-50 training with Triton-compiled custom GPU kernels.

Replaces standard PyTorch BatchNorm2d with a fused Triton BatchNorm+ReLU kernel
to demonstrate MPS partition sensitivity with custom GPU kernels.
Used for SoCC'26 rebuttal (reviewer question on Triton-compiled kernels).

Requires: PyTorch >= 2.0 (ships with triton)
"""

import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import triton
import triton.language as tl

# import custom functions
sys.path.append('../..')
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger
from utils.Dataset import DummyImageDataset


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _bn_relu_fwd_kernel(
    X_ptr, Y_ptr,
    W_ptr, B_ptr, Mean_ptr, Var_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused BatchNorm (using running stats) + ReLU forward – elementwise over NCHW tensor."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Channel index for NCHW layout: c = (flat_index / HW) % C
    c = (offs // HW) % C

    mean = tl.load(Mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(Var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(W_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + c, mask=mask, other=0.0).to(tl.float32)

    # BN + ReLU
    y = w * (x - mean) * tl.libdevice.rsqrt(var + eps) + b
    y = tl.maximum(y, 0.0)

    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def _relu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Standalone Triton ReLU kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Autograd wrappers (needed for training backward pass)
# ---------------------------------------------------------------------------

class TritonBNReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, eps, training, momentum):
        assert x.is_cuda, "Triton kernels require CUDA tensors"
        x = x.contiguous()
        N_batch, C, H, W = x.shape
        HW = H * W
        numel = x.numel()

        if training:
            # Compute batch statistics
            mean = x.float().mean(dim=(0, 2, 3))
            var = x.float().var(dim=(0, 2, 3), unbiased=False)
            # Update running stats
            with torch.no_grad():
                running_mean.mul_(1 - momentum).add_(mean * momentum)
                running_var.mul_(1 - momentum).add_(var * momentum)
        else:
            mean = running_mean
            var = running_var

        y = torch.empty_like(x)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _bn_relu_fwd_kernel[grid](
            x, y,
            weight, bias, mean, var,
            numel, C, HW, eps,
            BLOCK_SIZE,
        )

        ctx.save_for_backward(x, weight, mean, var)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, var = ctx.saved_tensors
        eps = ctx.eps
        N_batch, C, H, W = x.shape

        # Recompute normalized x and relu mask
        rstd = torch.rsqrt(var + eps)
        x_hat = (x.float() - mean.view(1, -1, 1, 1)) * rstd.view(1, -1, 1, 1)
        bn_out = weight.view(1, -1, 1, 1).float() * x_hat + ctx.saved_tensors[1].view(1, -1, 1, 1).float()  # bias
        # Recompute from saved: bn_out = weight * x_hat + bias
        bn_out_recomp = weight.float().view(1, -1, 1, 1) * x_hat
        # Actually just use the forward result logic: y = w*(x-mean)*rstd + b, relu_mask = y > 0
        y_pre_relu = weight.float().view(1, -1, 1, 1) * x_hat + ctx.saved_tensors[1].float().view(1, -1, 1, 1)
        relu_mask = (y_pre_relu > 0).float()

        dy_float = dy.float() * relu_mask

        # Gradients for BN
        dbias = dy_float.sum(dim=(0, 2, 3))
        dweight = (dy_float * x_hat).sum(dim=(0, 2, 3))

        # dx through BN
        M = N_batch * H * W
        dx_hat = dy_float * weight.float().view(1, -1, 1, 1)
        dx = rstd.view(1, -1, 1, 1) / M * (
            M * dx_hat
            - dx_hat.sum(dim=(0, 2, 3), keepdim=True)
            - x_hat * (dx_hat * x_hat).sum(dim=(0, 2, 3), keepdim=True)
        )

        return dx.to(dy.dtype), dweight.to(weight.dtype), dbias.to(weight.dtype), None, None, None, None, None


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------

class TritonFusedBNReLU(nn.Module):
    """Fused BatchNorm+ReLU using Triton kernel, supports training."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.weight = bn.weight
        self.bias = bn.bias
        self.register_buffer('running_mean', bn.running_mean)
        self.register_buffer('running_var', bn.running_var)
        self.eps = bn.eps
        self.momentum = bn.momentum
        self.num_features = bn.num_features

    def forward(self, x):
        return TritonBNReLUFn.apply(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.eps, self.training, self.momentum,
        )


class TritonReLU(nn.Module):
    """Drop-in replacement for nn.ReLU using Triton kernel."""

    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK_SIZE = 1024
        _relu_kernel[(triton.cdiv(N, BLOCK_SIZE),)](x, y, N, BLOCK_SIZE)
        return y


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def patch_resnet_with_triton(model):
    """Replace BatchNorm2d+ReLU pairs in ResNet-50 with fused Triton kernel.

    For ResNet, Conv blocks are Sequentials of [Conv2d, BatchNorm2d] followed
    by a ReLU. We fuse BN+ReLU into one Triton kernel where possible.
    """
    fused_count = 0
    relu_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            children = list(module.children())
            for i in range(len(children) - 1):
                if isinstance(children[i], nn.BatchNorm2d) and isinstance(children[i + 1], nn.ReLU):
                    fused = TritonFusedBNReLU(children[i])
                    module[i] = fused
                    module[i + 1] = nn.Identity()
                    fused_count += 1

    # Replace top-level relu in ResNet (model.relu)
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, name, TritonReLU())
            relu_count += 1

    return model, fused_count, relu_count


# ---------------------------------------------------------------------------
# Training loop (mirrors imgclassification-train.py)
# ---------------------------------------------------------------------------

def train(args, profiler=None):
    logger = MLLogger(args.log_dir, args, __file__)
    logger.log(f"args={args}")
    logger.log(f"[Triton] Custom-kernel ResNet-50 training")
    logger.log(f"using {args.device}...")

    run_epochs = args.n_epoch
    if args.profile:
        GPUinfo = []
        if not args.profile_all_estimation:
            run_epochs = min(args.n_epoch, args.profile_n_epoch)
            logger.log(f"[Profile] early stop at epoch{run_epochs}")
            profiler.profile_n_epoch = run_epochs

    # Dataset – ResNet-50 uses 32x32 dummy images (matching original imgclassification-train.py)
    train_dataset = DummyImageDataset(num_samples=10000, image_shape=(3, 32, 32))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Model – strip '-triton' suffix if present to get the base model name
    hf_model_name = args.model_name.replace('-triton', '')
    logger.log(f"Loading model: {hf_model_name}")
    model = models.resnet50(pretrained=True)

    # Patch with Triton custom kernels
    model, fused_count, relu_count = patch_resnet_with_triton(model)
    logger.log(f"[Triton] Fused {fused_count} BN+ReLU -> TritonFusedBNReLU")
    logger.log(f"[Triton] Replaced {relu_count} standalone ReLU -> TritonReLU")

    model.to(args.device)

    if args.profile:
        profiler.getModelMemory()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.wandb_logging:
        import wandb
        wandb.init(entity="nba556677go", project="k8s-scheduling", name=args.exp_name, config=args)
        wandb.watch(model)

    start_time = time.time()
    step_process_time = []
    warmup = 10
    steps = 0

    for epoch in range(1, run_epochs + 1):
        model.train()
        running_loss = 0.0
        start_epoch_time = time.time()

        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            if steps == warmup:
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"steps{steps}")

            step_time = time.time()

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"moveData{steps}")
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            optimizer.zero_grad()

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"forward{steps}")
            outputs = model(inputs)
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"backward{steps}")
            loss = criterion(outputs, labels)
            loss.backward()
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"optimize{steps}")
            optimizer.step()
            optimizer.zero_grad()
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            running_loss += loss.item()
            step_process_time.extend([time.time() - step_time] * args.batch_size)
            logger.log(f"average step time: {sum(step_process_time) / len(step_process_time):.4f} seconds")

            if args.profile_1step:
                logger.log(f'profile only 1 step for nightsight compute...')
                exit(0)
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()
            if args.profile_nstep > 0 and steps >= args.profile_nstep:
                logger.log(f"completed {args.profile_nstep} steps, exiting...")
                torch.cuda.cudart().cudaProfilerStop()
                exit(0)
            steps += 1

        epoch_time = time.time() - start_epoch_time
        logger.log(f"Epoch {epoch}, Loss: {running_loss / len(train_dataloader)}, Time: {epoch_time:.2f} seconds")

        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()
            GPUinfo = profiler.getSMIinfobyTask(sys.argv)
            logger.log(f"[Profile]GPU INFO - {GPUinfo}")
            gpuInfoDict = list(GPUinfo.values())[0][0]
            profiler.addEpochTime(epoch_time)

        if args.wandb_logging:
            wandb.log({"train_loss": running_loss / len(train_dataloader), "epoch": epoch})

    total_training_time = time.time() - start_time
    m, s = divmod(total_training_time, 60)
    h, m = divmod(m, 60)
    logger.log(f"Total Training Time: {h:f}h {m:02f}m {s:02f}s")
    training_time_per_epoch = total_training_time / run_epochs
    m, s = divmod(training_time_per_epoch, 60)
    h, m = divmod(m, 60)
    logger.log(f"Average Time for each Epoch: {h:f}h {m:02f}m {s:02f}s")

    if args.profile and not args.profile_all_estimation:
        profiler.saveEarlyStop(os.path.basename(__file__)[:-3], GPUinfo, training_time_per_epoch)
        logger.log(f"Earlystop at epoch{run_epochs}")

    logger.log("Training finished!")


if __name__ == "__main__":
    args = MLParser(mode="train").get_args()
    profiler = MLProfiler(args) if args.profile else None
    train(args, profiler)
