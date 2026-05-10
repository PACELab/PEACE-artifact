"""
MobileNet V2 inference with Triton-compiled custom GPU kernels.

Replaces standard PyTorch BatchNorm2d+ReLU6 with a fused Triton kernel
to demonstrate MPS partition sensitivity with custom GPU kernels.
Used for SoCC'26 rebuttal (reviewer question on Triton-compiled kernels).

Requires: PyTorch >= 2.0 (ships with triton)
"""

import os
import sys
import time
import argparse
import signal
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import triton
import triton.language as tl

# import custom functions
sys.path.append('../..')
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger


# ---------------------------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------------------------

@triton.jit
def _fused_bn_relu6_kernel(
    X_ptr, Y_ptr,
    W_ptr, B_ptr, Mean_ptr, Var_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused BatchNorm (inference) + ReLU6 – elementwise over NCHW tensor."""
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

    # Fused BN + ReLU6
    y = w * (x - mean) * tl.libdevice.rsqrt(var + eps) + b
    y = tl.minimum(tl.maximum(y, 0.0), 6.0)

    tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def _relu6_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Standalone Triton ReLU6 for layers where BN is not fused."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.minimum(tl.maximum(x, 0.0), 6.0)
    tl.store(Y_ptr + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------

class TritonFusedBNReLU6(nn.Module):
    """Fused BatchNorm+ReLU6 using Triton kernel (inference only)."""

    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.weight = bn.weight
        self.bias = bn.bias
        self.register_buffer('running_mean', bn.running_mean)
        self.register_buffer('running_var', bn.running_var)
        self.eps = bn.eps
        self.num_features = bn.num_features

    def forward(self, x):
        assert x.is_cuda, "Triton kernels require CUDA tensors"
        x = x.contiguous()
        N, C, H, W = x.shape
        HW = H * W
        numel = x.numel()
        y = torch.empty_like(x)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _fused_bn_relu6_kernel[grid](
            x, y,
            self.weight, self.bias, self.running_mean, self.running_var,
            numel, C, HW, self.eps,
            BLOCK_SIZE,
        )
        return y


class TritonReLU6(nn.Module):
    """Drop-in replacement for nn.ReLU6 using Triton kernel."""

    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        N = x.numel()
        BLOCK_SIZE = 1024
        _relu6_kernel[(triton.cdiv(N, BLOCK_SIZE),)](x, y, N, BLOCK_SIZE)
        return y


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def patch_mobilenet_with_triton(model):
    """Replace BatchNorm2d+ReLU6 pairs in Sequential blocks with fused Triton kernel.

    For MobileNet V2, Conv2dNormActivation blocks are Sequentials of
    [Conv2d, BatchNorm2d, ReLU6]. We fuse BN+ReLU6 into one Triton kernel
    and replace the ReLU6 with Identity.
    """
    fused_count = 0
    relu6_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            children = list(module.children())
            for i in range(len(children) - 1):
                if isinstance(children[i], nn.BatchNorm2d) and isinstance(children[i + 1], (nn.ReLU6, nn.Hardswish)):
                    fused = TritonFusedBNReLU6(children[i])
                    module[i] = fused
                    module[i + 1] = nn.Identity()
                    fused_count += 1

        # Also replace standalone ReLU6 modules not preceded by BN
        if isinstance(module, nn.ReLU6):
            # handled above via parent Sequential; skip standalone replacements
            pass

    # Replace any remaining standalone ReLU6 at top level
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(model, name, TritonReLU6())
            relu6_count += 1

    return model, fused_count, relu6_count


# ---------------------------------------------------------------------------
# Inference loop (mirrors imgclassification-inference.py)
# ---------------------------------------------------------------------------

def inference(args, profiler=None):
    logger = MLLogger(args.log_dir, args, __file__)
    logger.log(f"[Triton] Custom-kernel MobileNet V2 inference")
    logger.log(f"using {args.device}...")

    # Load MobileNet V2 from torchvision
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    # Patch with Triton custom kernels
    model, fused_count, relu6_count = patch_mobilenet_with_triton(model)
    logger.log(f"[Triton] Fused {fused_count} BN+ReLU6 -> TritonFusedBNReLU6")
    logger.log(f"[Triton] Replaced {relu6_count} standalone ReLU6 -> TritonReLU6")

    if "cuda" in str(args.device):
        model.to(args.device)

    # Generate dummy input matching CIFAR-10 resized for MobileNet (3x224x224)
    # We loop over batches of synthetic data to simulate continuous inference
    def generate_batch(batch_size):
        return torch.randn(batch_size, 3, 224, 224)

    num_batches = 10000  # large pool to iterate over

    def countAverageProcessingTime():
        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {req_total_time:.4f} seconds")
        logger.writecsv(
            f"req_device{args.device}batch{args.batch_size}_inference_process_time.csv",
            [avg_processing_time],
        )

    def sigterm_handler(signum, frame):
        countAverageProcessingTime()
        logger.log("completed nstep, exiting...")
        exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    req_processing_time, req_total_time = [], 0
    warmup = 10
    steps = 0

    while True:
        if steps < warmup:
            logger.log(f"warmup steps {steps}")
        if steps == warmup:
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()
        if steps >= warmup and args.profile:
            torch.cuda.nvtx.range_push(f"steps{steps}")

        if steps >= warmup and args.profile:
            torch.cuda.nvtx.range_push(f"moveData{steps}")
        images = generate_batch(args.batch_size).to(args.device)
        if steps >= warmup and args.profile:
            torch.cuda.nvtx.range_pop()

        start_time = time.time()

        if steps >= warmup and args.profile:
            torch.cuda.nvtx.range_push(f"forward{steps}")
        with torch.no_grad():
            outputs = model(images)
        if steps >= warmup and args.profile:
            torch.cuda.nvtx.range_pop()

        if args.profile_1step:
            logger.log("in 1step")
            exit(0)

        predictions = torch.argmax(outputs, dim=-1)
        logger.log(f"predictions: {predictions}")

        if steps >= warmup and args.profile:
            torch.cuda.nvtx.range_pop()

        elapsed = time.time() - start_time
        req_processing_time.extend([elapsed] * args.batch_size)
        req_total_time += elapsed

        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {req_total_time:.4f} seconds")

        if args.profile_nstep > 0 and steps >= args.profile_nstep:
            if steps >= warmup and args.profile:
                torch.cuda.cudart().cudaProfilerStop()
            countAverageProcessingTime()
            logger.log(f"completed {steps} steps, exiting...")
            exit(0)

        steps += 1

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    countAverageProcessingTime()


if __name__ == "__main__":
    args = MLParser(mode="inference").get_args()
    profiler = MLProfiler(args) if args.profile else None
    inference(args, profiler)
    exit(0)
