"""
Wav2Vec2 inference with Triton-compiled custom GPU kernels.

Replaces standard PyTorch LayerNorm and GELU with fused Triton kernels
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
from torch.utils.data import DataLoader, Dataset

import triton
import triton.language as tl

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# import custom functions
sys.path.append('../..')
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger


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
    cdf = 0.5 * (1.0 + tl.libdevice.tanh(
        0.7978845608028654 * (x + 0.044715 * x * x * x)
    ))
    y = x * cdf
    tl.store(Y + offs, y, mask=mask)


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
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        M, N = x_2d.shape

        y = torch.empty_like(x_2d)
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)

        BLOCK_SIZE = triton.next_power_of_2(N)
        _layer_norm_fwd_kernel[(M,)](
            x_2d, y, self.weight, self.bias, mean, rstd,
            x_2d.stride(0), N, self.eps,
            BLOCK_SIZE,
        )
        return y.reshape(orig_shape)


class TritonGELU(nn.Module):
    """Drop-in replacement for GELU using Triton forward kernel."""

    def forward(self, x):
        x_flat = x.contiguous().view(-1)
        N = x_flat.numel()
        y = torch.empty_like(x_flat)
        BLOCK_SIZE = 1024
        _gelu_fwd_kernel[(triton.cdiv(N, BLOCK_SIZE),)](x_flat, y, N, BLOCK_SIZE)
        return y.view_as(x)


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
    """Replace GELU activation functions with Triton GELU."""
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, TritonGELU())
        # Wav2Vec2 stores activation as a module attribute in feed-forward layers
        if hasattr(child, 'activation') and not isinstance(child.activation, TritonGELU):
            if isinstance(child.activation, nn.GELU):
                child.activation = TritonGELU()
        _replace_gelu_activations(child)


def patch_wav2vec_with_triton(model):
    """Patch a Wav2Vec2 model to use Triton custom kernels."""
    _replace_layernorms(model)
    _replace_gelu_activations(model)
    triton_ln_count = sum(1 for m in model.modules() if isinstance(m, TritonLayerNorm))
    triton_gelu_count = sum(1 for m in model.modules() if isinstance(m, TritonGELU))
    return model, triton_ln_count, triton_gelu_count


# ---------------------------------------------------------------------------
# Dummy dataset (matches speech-recognition-inference.py)
# ---------------------------------------------------------------------------

class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_length, num_channels=0):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_channels = num_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.num_channels > 0:
            return torch.randn(self.num_channels, self.seq_length)
        else:
            return torch.randn(self.seq_length)


def collate_fn(batch):
    batch = [item.squeeze(0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch


# ---------------------------------------------------------------------------
# Inference loop (mirrors speech-recognition-inference.py)
# ---------------------------------------------------------------------------

def inference(args, profiler=None):
    logger = MLLogger(args.log_dir, args, __file__)
    logger.log(f"args={args}")
    logger.log(f"[Triton] Custom-kernel Wav2Vec2 inference")
    logger.log(f"using {args.device}...")

    # Load Wav2Vec2 – strip '-triton' suffix if present
    hf_model_name = args.model_name.replace('-triton', '')
    logger.log(f"Loading HuggingFace model: {hf_model_name}")

    processor = Wav2Vec2Processor.from_pretrained(hf_model_name)
    model = Wav2Vec2ForCTC.from_pretrained(hf_model_name)
    model.eval()

    # Patch with Triton custom kernels
    model, ln_count, gelu_count = patch_wav2vec_with_triton(model)
    logger.log(f"[Triton] Patched {ln_count} LayerNorm -> TritonLayerNorm")
    logger.log(f"[Triton] Patched {gelu_count} GELU -> TritonGELU")

    model.to(args.device)

    # Dummy dataset matching wav2vec input shape
    dummy_dataset = DummyDataset(num_samples=1000000000000, seq_length=160487, num_channels=0)
    data_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    req_processing_time = []
    warmup = 10

    def countAverageProcessingTime():
        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {avg_processing_time * len(req_processing_time):.4f} seconds")
        logger.writecsv(
            f"req_device{args.device}batch{args.batch_size}_inference_process_time.csv",
            [avg_processing_time],
        )

    def sigterm_handler(signum, frame):
        countAverageProcessingTime()
        logger.log("completed nstep, exiting...")
        exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    for steps, batch in enumerate(data_loader):
        if steps == warmup:
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()

        if steps >= warmup and args.profile:
            torch.cuda.nvtx.range_push(f"steps{steps}")
            torch.cuda.nvtx.range_push(f"moveData{steps}")

        start_time = time.time()
        batch = batch.to(args.device)

        if steps >= warmup and args.profile:
            torch.cuda.nvtx.range_pop()

        if steps >= warmup:
            if args.profile:
                torch.cuda.nvtx.range_push(f"forward{steps}")
            with torch.no_grad():
                outputs = model(batch).logits
            if args.profile:
                torch.cuda.nvtx.range_pop()

            predicted_ids = torch.argmax(outputs, dim=-1)

            if args.profile:
                torch.cuda.nvtx.range_push(f"decode{steps}")
            transcriptions = processor.batch_decode(predicted_ids)
            if args.profile:
                torch.cuda.nvtx.range_pop()

            print(transcriptions)

            end_time = time.time()
            req_processing_time.extend([end_time - start_time] * args.batch_size)
            countAverageProcessingTime()

            if args.profile:
                torch.cuda.nvtx.range_pop()

            if args.profile_nstep > 0 and steps >= args.profile_nstep:
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStop()
                logger.log(f"completed {steps} steps, exiting...")
                logger.writecsv(
                    f"req_device{args.device}_batch{args.batch_size}__inference_process_time.csv",
                    req_processing_time,
                )
                exit(0)
        else:
            logger.log(f"warmup step {steps}")
            with torch.no_grad():
                outputs = model(batch).logits
            predicted_ids = torch.argmax(outputs, dim=-1)
            transcriptions = processor.batch_decode(predicted_ids)
            end_time = time.time()
            logger.log(f"Average processing time: {end_time - start_time:.4f} seconds")

        if args.profile_1step:
            print("in 1step")
            exit(0)

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    logger.writecsv(
        f"req_device{args.device}_batch{args.batch_size}__inference_process_time.csv",
        req_processing_time,
    )


if __name__ == "__main__":
    args = MLParser(mode="inference").get_args()
    profiler = MLProfiler(args) if args.profile else None
    inference(args, profiler)
    exit(0)
