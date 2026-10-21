"""Attention kernels with a dependency-free numerical model."""

from .online import dense_attention, online_attention

__version__ = "0.1.0"
__all__ = ["attention", "reference_attention", "dense_attention", "online_attention"]


def __getattr__(name):
    if name in ("attention", "reference_attention"):
        from . import torch_ops

        return getattr(torch_ops, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
