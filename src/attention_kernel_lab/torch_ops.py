"""Dense PyTorch oracle and explicit dispatch for forward attention.

Importing this module does not import PyTorch or Triton. The GPU implementation
is deliberately inference-only; the reference remains differentiable.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import numbers
import sys
from typing import Any

SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)
SUPPORTED_BLOCKS = (16, 32, 64)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _torch() -> Any:
    if not torch_available():
        raise RuntimeError("PyTorch is required for tensor attention; install the torch extra.")
    return importlib.import_module("torch")


def _validate(q: Any, k: Any, v: Any, causal: bool, scale: float | None) -> tuple[Any, float]:
    torch = _torch()
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v)):
        raise TypeError("q, k, v must be PyTorch tensors")
    if any(x.ndim != 4 for x in (q, k, v)):
        raise ValueError("expected q, k, v with layout [batch, heads, sequence, head_dim]")
    if any(any(size <= 0 for size in x.shape) for x in (q, k, v)):
        raise ValueError("all tensor dimensions must be positive")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("batch and head counts must match; GQA and broadcasting are unsupported")
    if k.shape[2] != v.shape[2] or q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("K/V sequence lengths and all head dimensions must match")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    if q.device.type not in ("cpu", "cuda"):
        raise ValueError("the reference targets CPU and CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("expected float16, bfloat16, float32, or float64 tensors")
    if not all(x.is_contiguous() for x in (q, k, v)):
        raise ValueError("q, k, v must be contiguous in [B,H,N,D] layout")
    if type(causal) is not bool:
        raise TypeError("causal must be bool")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
        raise TypeError("scale must be a real number or None")
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return torch, float(scale)


def reference_attention(
    q: Any, k: Any, v: Any, *, causal: bool = False, scale: float | None = None
) -> Any:
    """Materialize softmax(Q K^T * scale) V as a correctness oracle.

    Non-square causal attention uses the upper-left mask ``key_index <=
    query_index``. Inputs must be finite; NaN/Inf checking is left to callers
    so dispatch does not add a device synchronization. Low-precision inputs
    accumulate in float32, float64 inputs retain float64, output keeps q.dtype.
    For a CUDA oracle, disable process-wide TF32 matmul before calling (the
    benchmark does this). This function does not mutate global backend policy.
    This implementation allocates O(B*H*Nq*Nk) scores.
    """
    torch, effective_scale = _validate(q, k, v, causal, scale)
    accum_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    # Outer autocast must not quietly lower the oracle's accumulation precision.
    with torch.autocast(device_type=q.device.type, enabled=False):
        scores = (q.to(accum_dtype) @ k.to(accum_dtype).transpose(-2, -1)) * effective_scale
        if causal:
            rows = torch.arange(q.shape[2], device=q.device)[:, None]
            cols = torch.arange(k.shape[2], device=q.device)[None, :]
            scores = scores.masked_fill(cols > rows, float("-inf"))
        return (torch.softmax(scores, dim=-1) @ v.to(accum_dtype)).to(q.dtype)


def _triton_reason(torch: Any, q: Any, k: Any, v: Any) -> str | None:
    if q.device.type != "cuda":
        return "Triton backend requires CUDA tensors"
    if torch.version.hip is not None:
        return "this project's Triton backend does not target ROCm"
    if not sys.platform.startswith("linux"):
        return "this project's Triton backend targets Linux (including a configured WSL2 environment)"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return "Triton backend requires float16 or bfloat16 inputs"
    if q.shape[-1] not in SUPPORTED_HEAD_DIMS:
        return f"Triton head_dim must be one of {SUPPORTED_HEAD_DIMS}"
    if any(x.numel() >= 2**31 for x in (q, k, v)):
        return "Triton backend requires each input to contain fewer than 2**31 elements"
    if q.shape[0] * q.shape[1] > 65535:
        return "Triton backend requires batch * heads <= 65535"
    if any(x.requires_grad for x in (q, k, v)):
        return "Triton forward is inference-only; requires_grad inputs are unsupported"
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        return "Triton backend requires NVIDIA compute capability 8.0 or newer"
    if importlib.util.find_spec("triton") is None:
        return "Triton is not installed"
    return None


def triton_unavailable_reason(q: Any, k: Any, v: Any) -> str | None:
    """Return the eligibility reason, without launching or compiling a kernel.

    ``None`` means the input passes this project's checks, not that the hardware
    has been validated. Installed-but-broken runtime errors are not suppressed.
    """
    torch, _ = _validate(q, k, v, False, None)
    return _triton_reason(torch, q, k, v)


def attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    causal: bool = False,
    scale: float | None = None,
    backend: str = "auto",
    block_m: int = 32,
    block_n: int = 32,
) -> Any:
    """Dispatch to ``reference`` or ``triton``; ``auto`` checks eligibility.

    An eligible Triton launch failure always propagates, even in auto mode.
    Choose ``reference`` explicitly when a differentiable implementation is
    needed. No strides, dtype conversions, or device copies are hidden here.
    """
    torch, effective_scale = _validate(q, k, v, causal, scale)
    if backend not in ("auto", "reference", "triton"):
        raise ValueError("backend must be 'auto', 'reference', or 'triton'")
    if type(block_m) is not int or type(block_n) is not int:
        raise TypeError("block_m and block_n must be integers")
    if block_m not in SUPPORTED_BLOCKS or block_n not in SUPPORTED_BLOCKS:
        raise ValueError(f"block_m and block_n must be one of {SUPPORTED_BLOCKS}")
    if backend == "reference":
        return reference_attention(q, k, v, causal=causal, scale=effective_scale)
    reason = _triton_reason(torch, q, k, v)
    if reason is not None:
        if backend == "triton":
            raise RuntimeError(reason)
        return reference_attention(q, k, v, causal=causal, scale=effective_scale)
    kernel = importlib.import_module("attention_kernel_lab.triton_kernel")
    return kernel.launch_attention(q, k, v, causal, effective_scale, block_m, block_n)
