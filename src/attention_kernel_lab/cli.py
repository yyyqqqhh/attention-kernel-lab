"""Command-line experiments. Framework dependencies are imported on demand."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from .benchmark import BenchmarkConfig, error_metrics, run_benchmark
from .environment import collect_environment


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _shape_options(parser, *, bench: bool = False) -> None:
    parser.add_argument("--batch", type=_positive_int, default=1)
    parser.add_argument("--heads", type=_positive_int, default=4 if bench else 2)
    parser.add_argument("--nq", type=_positive_int, default=512 if bench else 33)
    parser.add_argument("--nk", type=_positive_int, default=512 if bench else 65)
    parser.add_argument("--dim", type=_positive_int, default=64 if bench else 32)
    parser.add_argument("--dtype", choices=("fp16", "bf16") if bench else ("fp16", "bf16", "fp32"), default="fp16" if bench else "fp32")
    parser.add_argument("--causal", action="store_true", help="upper-left mask: key index <= query index")
    parser.add_argument("--scale", type=_finite_float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-m", type=_positive_int, default=32)
    parser.add_argument("--block-n", type=_positive_int, default=32)
    parser.add_argument("--output", type=Path, help="save the report as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="attention-kernel-lab", description="Reproducible forward attention experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="read-only environment inventory")
    doctor.add_argument("--json", action="store_true", help="retained for explicit machine-readable invocation; all reports are JSON")
    doctor.add_argument("--output", type=Path)
    demo = commands.add_parser("demo", help="compare dense and online softmax with standard-library Python")
    demo.add_argument("--nq", type=_positive_int, default=7)
    demo.add_argument("--nk", type=_positive_int, default=11)
    demo.add_argument("--dim", type=_positive_int, default=4)
    demo.add_argument("--block-size", type=_positive_int, default=3)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--causal", action="store_true")
    demo.add_argument("--scale", type=_finite_float, default=None)
    demo.add_argument("--output", type=Path)
    check = commands.add_parser("check", help="compare PyTorch/Triton output against eager reference")
    _shape_options(check)
    check.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    check.add_argument("--backend", choices=("reference", "triton", "auto"), default="reference")
    bench = commands.add_parser("bench", help="correctness-gated CUDA-event measurements")
    _shape_options(bench, bench=True)
    bench.add_argument("--warmup", type=_positive_int, default=10)
    bench.add_argument("--samples", type=_positive_int, default=30)
    bench.add_argument("--iterations", type=_positive_int, default=10)
    bench.add_argument("--include-eager", action="store_true")
    bench.add_argument("--include-math", action="store_true")
    return parser


def _demo(args) -> dict:
    from .online import dense_attention, online_attention

    if args.nq * args.nk * args.dim > 2_000_000:
        raise ValueError("stdlib demo is limited to nq*nk*dim <= 2,000,000; use check for larger tensors")
    generator = random.Random(args.seed)
    q = [[generator.uniform(-2, 2) for _ in range(args.dim)] for _ in range(args.nq)]
    k = [[generator.uniform(-2, 2) for _ in range(args.dim)] for _ in range(args.nk)]
    v = [[generator.uniform(-2, 2) for _ in range(args.dim)] for _ in range(args.nk)]
    kwargs = {"causal": args.causal, "scale": args.scale}
    expected = dense_attention(q, k, v, **kwargs)
    actual = online_attention(q, k, v, block_size=args.block_size, **kwargs)
    max_abs = max(abs(x - y) for row, reference_row in zip(actual, expected) for x, y in zip(row, reference_row))
    if max_abs > 1e-10:
        raise RuntimeError(f"online softmax mismatch: max_abs={max_abs}")
    return {
        "status": "passed",
        "implementation": "stdlib Python double precision; not a GPU performance result",
        "shape": {"nq": args.nq, "nk": args.nk, "dim": args.dim},
        "causal": args.causal,
        "causal_alignment": "upper-left (j <= i)",
        "scale": args.scale,
        "effective_scale": args.scale if args.scale is not None else 1.0 / math.sqrt(args.dim),
        "block_size": args.block_size,
        "seed": args.seed,
        "max_abs_error": max_abs,
        "tolerance": 1e-10,
        "output_preview": actual[:2],
    }


def _check(args) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable; install the documented CPU or GPU environment for check") from exc
    from .torch_ops import attention, reference_attention

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no GPU correctness check was run")
    if args.device == "cuda" and torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("FP32 correctness oracle requires torch.backends.cuda.matmul.allow_tf32=False")
    if args.batch * args.heads * args.nq * args.nk > 16_777_216:
        raise ValueError("dense correctness reference is limited to 16,777,216 score elements")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    q = torch.randn((args.batch, args.heads, args.nq, args.dim), dtype=dtype, device=args.device, generator=generator)
    k = torch.randn((args.batch, args.heads, args.nk, args.dim), dtype=dtype, device=args.device, generator=generator)
    v = torch.randn(k.shape, dtype=dtype, device=args.device, generator=generator)
    with torch.inference_mode():
        actual = attention(q, k, v, causal=args.causal, scale=args.scale, backend=args.backend, block_m=args.block_m, block_n=args.block_n)
        expected = reference_attention(q, k, v, causal=args.causal, scale=args.scale)
        from torch.nn import functional as functional

        sdpa = functional.scaled_dot_product_attention(q, k, v, is_causal=args.causal, scale=args.scale, dropout_p=0.0)
        atol, rtol = {"fp16": (2e-2, 2e-2), "bf16": (5e-2, 5e-2), "fp32": (2e-5, 2e-5)}[args.dtype]
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
        torch.testing.assert_close(sdpa.float(), expected.float(), atol=atol, rtol=rtol)
        metrics = {"requested_backend": error_metrics(actual, expected), "sdpa_auto": error_metrics(sdpa, expected)}
    return {
        "status": "passed",
        "requested_backend": args.backend,
        "dispatch_note": "auto may choose reference; only explicit triton establishes a GPU kernel check",
        "device": args.device,
        "dtype": args.dtype,
        "shape": {name: getattr(args, name) for name in ("batch", "heads", "nq", "nk", "dim")},
        "causal": args.causal,
        "scale": args.scale,
        "effective_scale": args.scale if args.scale is not None else 1.0 / math.sqrt(args.dim),
        "seed": args.seed,
        "tolerance": {"atol": atol, "rtol": rtol},
        "reference": "eager_fp32_scores_output_cast_to_input_dtype",
        "errors": metrics,
        "environment": collect_environment(),
    }


def _report(result: dict, output: Path | None) -> None:
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = collect_environment()
        elif args.command == "demo":
            result = _demo(args)
        elif args.command == "check":
            result = _check(args)
        else:
            config_fields = BenchmarkConfig.__dataclass_fields__
            result = run_benchmark(BenchmarkConfig(**{name: getattr(args, name) for name in config_fields}))
        _report(result, args.output)
    except (RuntimeError, ValueError, ImportError, OSError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
