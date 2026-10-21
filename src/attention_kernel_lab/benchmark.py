"""Correctness-gated CUDA-event benchmarks, with optional imports kept lazy."""

from __future__ import annotations

import importlib.util
import math
import numbers
import platform
import statistics
from dataclasses import asdict, dataclass

from .environment import collect_environment


@dataclass(frozen=True)
class BenchmarkConfig:
    batch: int = 1
    heads: int = 4
    nq: int = 512
    nk: int = 512
    dim: int = 64
    dtype: str = "fp16"
    causal: bool = False
    scale: float | None = None
    seed: int = 0
    warmup: int = 10
    samples: int = 30
    iterations: int = 10
    include_eager: bool = False
    include_math: bool = False
    block_m: int = 32
    block_n: int = 32

    def validate(self) -> None:
        for name in ("batch", "heads", "nq", "nk", "dim", "warmup", "samples", "iterations", "block_m", "block_n", "seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer, excluding bool")
        for name in ("causal", "include_eager", "include_math"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        for name in ("batch", "heads", "nq", "nk", "samples", "iterations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not -(2**63) <= self.seed <= 2**64 - 1:
            raise ValueError("seed must fit PyTorch's signed/unsigned 64-bit seed range")
        if self.warmup < 1:
            raise ValueError("warmup must be >= 1 to exclude first-call JIT compilation")
        if self.dtype not in ("fp16", "bf16"):
            raise ValueError("GPU benchmark dtype must be fp16 or bf16")
        if self.dim not in (16, 32, 64, 128):
            raise ValueError("GPU benchmark dim must be one of 16, 32, 64, 128")
        if self.scale is not None:
            if isinstance(self.scale, bool) or not isinstance(self.scale, numbers.Real):
                raise ValueError("scale must be a real number, excluding bool")
            if not math.isfinite(self.scale) or self.scale <= 0:
                raise ValueError("scale must be finite and positive")
        if self.block_m not in (16, 32, 64) or self.block_n not in (16, 32, 64):
            raise ValueError("block_m and block_n must be one of 16, 32, 64")
        # Correctness validation intentionally materializes a FP32 score matrix.
        # Bound it before allocating: this is an experiment harness, not serving.
        if self.batch * self.heads * self.nq * self.nk > 16_777_216:
            raise ValueError("dense correctness reference is limited to 16,777,216 score elements")
        if self.batch * self.heads > 65535:
            raise ValueError("batch * heads must be <= 65535")
        if self.batch * self.heads * (self.nq + 2 * self.nk) * self.dim > 134_217_728:
            raise ValueError("benchmark inputs are limited to 134,217,728 total elements")


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation between order statistics (fraction in [0, 1])."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires samples and a fraction in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def logical_flops(batch: int, heads: int, nq: int, nk: int, dim: int, causal: bool) -> int:
    """Logical QK and PV multiply-add work; excludes softmax and tile padding."""
    if causal:
        diagonal_rows = min(nq, nk)
        pairs = diagonal_rows * (diagonal_rows + 1) // 2 + max(nq - nk, 0) * nk
    else:
        pairs = nq * nk
    return 4 * batch * heads * pairs * dim


def error_metrics(actual, expected) -> dict:
    """Use FP32 differences; the relative denominator is explicitly floored."""
    import torch

    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_rel_denominator_floor_1e-3": (delta / expected.float().abs().clamp_min(1e-3)).max().item(),
        "all_finite": bool(torch.isfinite(actual).all().item()),
    }


def _require_cuda():
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is unavailable; install the documented GPU environment to benchmark")
    if platform.system() != "Linux":
        raise RuntimeError("Triton GPU benchmarks target Linux; native Windows is not supported")
    import torch

    if torch.version.hip:
        raise RuntimeError("This experiment targets NVIDIA CUDA; ROCm is not validated")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no GPU benchmark was run")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("The intended GPU target requires NVIDIA compute capability >= 8.0")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError("Triton is unavailable; install the documented GPU environment")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("FP32 correctness oracle requires torch.backends.cuda.matmul.allow_tf32=False; no benchmark was run")
    return torch


def _time_cuda(torch, operation, *, warmup: int, samples: int, iterations: int) -> dict:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) / iterations)
    median = statistics.median(elapsed)
    if median <= 0:
        raise RuntimeError("CUDA-event timer resolution was insufficient; increase --iterations")
    return {
        "median_ms": median,
        "p20_ms": percentile(elapsed, 0.2),
        "p80_ms": percentile(elapsed, 0.8),
        "sample_ms": elapsed,
    }


def run_benchmark(config: BenchmarkConfig) -> dict:
    """Verify every provider against eager FP32 scores before timing any provider."""
    config.validate()
    torch = _require_cuda()
    from torch.nn import functional as functional

    from .torch_ops import attention, reference_attention

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[config.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this CUDA device/runtime")
    generator = torch.Generator(device="cuda").manual_seed(config.seed)
    q = torch.randn((config.batch, config.heads, config.nq, config.dim), device="cuda", dtype=dtype, generator=generator)
    k = torch.randn((config.batch, config.heads, config.nk, config.dim), device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(k.shape, device="cuda", dtype=dtype, generator=generator)
    kwargs = {"causal": config.causal, "scale": config.scale}
    operations = {
        "triton": lambda: attention(q, k, v, backend="triton", block_m=config.block_m, block_n=config.block_n, **kwargs),
        "sdpa_auto": lambda: functional.scaled_dot_product_attention(q, k, v, is_causal=config.causal, scale=config.scale, dropout_p=0.0),
    }
    if config.include_math:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def math_operation():
            with sdpa_kernel(SDPBackend.MATH):
                return functional.scaled_dot_product_attention(q, k, v, is_causal=config.causal, scale=config.scale, dropout_p=0.0)

        operations["sdpa_math"] = math_operation
    if config.include_eager:
        operations["eager_fp32_scores"] = lambda: reference_attention(q, k, v, **kwargs)
    atol, rtol = ((2e-2, 2e-2) if config.dtype == "fp16" else (5e-2, 5e-2))
    with torch.inference_mode():
        expected = reference_attention(q, k, v, **kwargs)
        correctness = {}
        for name, operation in operations.items():
            output = operation()
            correctness[name] = error_metrics(output, expected)
            torch.testing.assert_close(output.float(), expected.float(), atol=atol, rtol=rtol)
            if not correctness[name]["all_finite"]:
                raise RuntimeError(f"{name} returned non-finite output; benchmark aborted")
        torch.cuda.synchronize()
        timings = {
            name: _time_cuda(torch, operation, warmup=config.warmup, samples=config.samples, iterations=config.iterations)
            for name, operation in operations.items()
        }
    flops = logical_flops(config.batch, config.heads, config.nq, config.nk, config.dim, config.causal)
    for timing in timings.values():
        timing["logical_tflops_per_second"] = flops / (timing["median_ms"] * 1e9)
    return {
        "schema_version": 1,
        "status": "measured",
        "config": asdict(config),
        "effective_scale": config.scale if config.scale is not None else 1.0 / math.sqrt(config.dim),
        "environment": collect_environment(),
        "device_index": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "correctness": {"atol": atol, "rtol": rtol, "reference": "eager_fp32_scores_output_cast_to_input_dtype", "providers": correctness},
        "timings": timings,
        "triton_speedup_over_sdpa_auto": timings["sdpa_auto"]["median_ms"] / timings["triton"]["median_ms"],
        "measurement": {
            "timer": "CUDA events on current stream; per-sample events synchronized",
            "scope": "reused preallocated inputs; each call allocates its output; no input transfer; no backward",
            "cache": "hot repeated same input buffers; no buffer rotation or cache flush; not a cold-DRAM measurement",
            "jit": "correctness calls and warmup occur before measured events",
            "providers": "sdpa_auto uses PyTorch dispatch; selected fused backend is not claimed",
            "reference_precision": "CUDA matmul TF32 must be disabled; the harness refuses enabled TF32 without changing process policy",
            "flops": "4*B*H*allowed_query_key_pairs*D; softmax and masked tile work excluded",
            "ordering": "providers timed sequentially; clock, thermal state, and other GPU work can affect results",
        },
    }
