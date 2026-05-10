"""
Albert training with Triton-compiled custom GPU kernels.

Replaces standard PyTorch LayerNorm and GELU with fused Triton kernels
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

import triton
import triton.language as tl

from transformers import AutoModelForSequenceClassification, get_scheduler

# import custom functions
sys.path.append('../..')
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger
from utils.Dataset import DummyBertDataset


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _layer_norm_fwd_kernel(
    X, Y, W, B, Mean, Rstd,
    stride,   # row stride of X
    N,        # number of columns (hidden_size)
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused LayerNorm forward – one program per row."""
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load row
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean
    mean = tl.sum(x, axis=0) / N

    # Variance
    xmean = x - mean
    var = tl.sum(xmean * xmean, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize + affine
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xmean * rstd * w + b

    tl.store(Y + cols, y, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


@triton.jit
def _gelu_fwd_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    """Fused GELU forward (tanh approximation)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.libdevice.tanh(
        0.7978845608028654 * (x + 0.044715 * x * x * x)
    ))
    y = x * cdf
    tl.store(Y + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Autograd wrappers
# ---------------------------------------------------------------------------

class TritonLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        M, N = x_2d.shape

        y = torch.empty_like(x_2d)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = triton.next_power_of_2(N)
        _layer_norm_fwd_kernel[(M,)](
            x_2d, y, weight, bias, mean, rstd,
            x_2d.stride(0), N, eps,
            BLOCK_SIZE,
        )

        ctx.save_for_backward(x_2d, weight, mean, rstd)
        ctx.N = N
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensors
        N = ctx.N
        dy_2d = dy.reshape(-1, N).contiguous()

        # Recompute x_hat
        x_hat = (x.float() - mean.unsqueeze(1)) * rstd.unsqueeze(1)

        dbias = dy_2d.float().sum(0)
        dweight = (dy_2d.float() * x_hat).sum(0)

        dx_hat = dy_2d.float() * weight.float()
        dx = rstd.unsqueeze(1) * (
            dx_hat
            - dx_hat.mean(-1, keepdim=True)
            - x_hat * (dx_hat * x_hat).mean(-1, keepdim=True)
        )
        return dx.to(dy.dtype).reshape(dy.shape), dweight.to(weight.dtype), dbias.to(weight.dtype), None


class TritonGELUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_flat = x.contiguous().view(-1)
        N = x_flat.numel()
        y = torch.empty_like(x_flat)
        BLOCK_SIZE = 1024
        _gelu_fwd_kernel[(triton.cdiv(N, BLOCK_SIZE),)](x_flat, y, N, BLOCK_SIZE)
        ctx.save_for_backward(x)
        return y.view_as(x)

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        xf = x.float()
        inner = 0.7978845608028654 * (xf + 0.044715 * xf ** 3)
        tanh_inner = torch.tanh(inner)
        cdf = 0.5 * (1.0 + tanh_inner)
        pdf = 0.5 * (1.0 - tanh_inner ** 2) * 0.7978845608028654 * (1.0 + 3 * 0.044715 * xf ** 2)
        return (dy.float() * (cdf + xf * pdf)).to(dy.dtype)


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------

class TritonLayerNorm(nn.Module):
    """Drop-in replacement for nn.LayerNorm using Triton forward kernel."""

    def __init__(self, orig: nn.LayerNorm):
        super().__init__()
        self.weight = orig.weight
        self.bias = orig.bias
        self.eps = orig.eps
        self.normalized_shape = orig.normalized_shape

    def forward(self, x):
        return TritonLayerNormFn.apply(x, self.weight, self.bias, self.eps)


class TritonGELU(nn.Module):
    """Drop-in replacement for GELU using Triton forward kernel."""

    def forward(self, x):
        return TritonGELUFn.apply(x)


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def _replace_layernorms(module):
    """Recursively replace nn.LayerNorm with TritonLayerNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TritonLayerNorm(child))
        else:
            _replace_layernorms(child)


def _replace_gelu_activations(module):
    """Replace GELU activation functions in Albert layers with Triton GELU."""
    for name, child in module.named_children():
        # Albert stores activation as module attribute 'activation'
        if hasattr(child, 'activation') and not isinstance(child.activation, TritonGELU):
            child.activation = TritonGELU()
        _replace_gelu_activations(child)


def patch_model_with_triton(model):
    """Patch an Albert model to use Triton custom kernels."""
    _replace_layernorms(model)
    _replace_gelu_activations(model)
    return model


# ---------------------------------------------------------------------------
# Training loop (mirrors recommend-train.py)
# ---------------------------------------------------------------------------

def train(args, profiler=None):
    logger = MLLogger(args.log_dir, args, __file__)
    logger.log(f"args={args}")
    logger.log(f"[Triton] Custom-kernel Albert training")
    logger.log(f"using {args.device}...")

    run_epochs = args.n_epoch
    if args.profile:
        GPUinfo = []
        if not args.profile_all_estimation:
            run_epochs = min(args.n_epoch, args.profile_n_epoch)
            logger.log(f"[Profile] early stop at epoch{run_epochs}")
            profiler.profile_n_epoch = run_epochs

    # Dataset
    vocab_size = 30522
    dummy_dataset = DummyBertDataset(num_samples=10000, seq_length=512, vocab_size=vocab_size)
    train_dataloader = DataLoader(dummy_dataset, shuffle=True, batch_size=args.batch_size)

    # Model – strip '-triton' suffix if present to get the HuggingFace model name
    hf_model_name = args.model_name.replace('-triton', '')
    logger.log(f"Loading HuggingFace model: {hf_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_name, num_labels=5)

    # Patch with Triton custom kernels
    model = patch_model_with_triton(model)
    triton_ln_count = sum(1 for m in model.modules() if isinstance(m, TritonLayerNorm))
    triton_gelu_count = sum(1 for m in model.modules() if isinstance(m, TritonGELU))
    logger.log(f"[Triton] Patched {triton_ln_count} LayerNorm -> TritonLayerNorm")
    logger.log(f"[Triton] Patched {triton_gelu_count} GELU -> TritonGELU")

    model.to(args.device)

    if args.profile:
        profiler.getModelMemory()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.n_epoch * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    if args.wandb_logging:
        import wandb
        wandb.init(entity="nba556677go", project="k8s-scheduling", name=args.exp_name, config=args)
        wandb.watch(model)

    progress_bar = tqdm(range(num_training_steps))

    start_time = time.time()
    step_process_time = []
    warmup = 10
    steps = 0
    model.train()

    for epoch in range(1, run_epochs + 1):
        start_epoch_time = time.time()
        running_loss = 0.0

        for batch in train_dataloader:
            if steps == warmup:
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"steps{steps}")

            step_time = time.time()

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"moveData{steps}")
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"forward{steps}")
            outputs = model(**batch)
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            loss = outputs.loss

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"backward{steps}")
            loss.backward()
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"optimize{steps}")
            optimizer.step()
            optimizer.zero_grad()
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_push(f"schedule{steps}")
            lr_scheduler.step()
            if steps >= warmup and args.profile:
                torch.cuda.nvtx.range_pop()

            progress_bar.update(1)
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
