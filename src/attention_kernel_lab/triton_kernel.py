"""NVIDIA/Linux tiled online-softmax forward kernel, loaded only on demand.

The recurrence follows the online softmax formulation used in FlashAttention.
This is a compact independent implementation, not a performance claim or a
drop-in training operator. See docs/design.md for the memory and layout tradeoffs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_fwd(
    Q, K, V, O,
    N_Q: tl.constexpr,
    N_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    cols_base = tl.arange(0, BLOCK_N)
    q = tl.load(
        Q + batch_head * N_Q * HEAD_DIM + rows[:, None] * HEAD_DIM + dims[None, :],
        mask=rows[:, None] < N_Q,
        other=0.0,
    )
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    end_n = N_K
    if CAUSAL:
        # Future K/V tiles cannot contribute to any row in this query tile.
        end_n = tl.minimum(N_K, (tile_m + 1) * BLOCK_M)
    for start_n in tl.range(0, end_n, BLOCK_N):
        cols = start_n + cols_base
        k = tl.load(
            K + batch_head * N_K * HEAD_DIM + cols[None, :] * HEAD_DIM + dims[:, None],
            mask=cols[None, :] < N_K,
            other=0.0,
        )
        scores = tl.dot(q, k) * SCALE
        valid = cols[None, :] < N_K
        if CAUSAL:
            valid = valid & (cols[None, :] <= rows[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        alpha = tl.exp(row_max - new_max)
        weights = tl.exp(scores - new_max[:, None])
        row_sum = row_sum * alpha + tl.sum(weights, axis=1)
        values = tl.load(
            V + batch_head * N_K * HEAD_DIM + cols[:, None] * HEAD_DIM + dims[None, :],
            mask=cols[:, None] < N_K,
            other=0.0,
        )
        # Tensor-core operands use input precision; statistics and output sums
        # stay FP32. This cast is part of the measured numerical error budget.
        acc = acc * alpha[:, None] + tl.dot(weights.to(values.dtype), values)
        row_max = new_max
    out = acc / row_sum[:, None]
    tl.store(
        O + batch_head * N_Q * HEAD_DIM + rows[:, None] * HEAD_DIM + dims[None, :],
        out,
        mask=rows[:, None] < N_Q,
    )


def launch_attention(q, k, v, causal, scale, block_m, block_n):
    """Launch after torch_ops has checked the public tensor contract."""
    out = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], block_m), q.shape[0] * q.shape[1])
    # Select the input device so a caller's unrelated current CUDA device cannot
    # place compilation/launch on a different GPU.
    with torch.cuda.device(q.device):
        _attention_fwd[grid](
            q, k, v, out,
            N_Q=q.shape[2], N_K=k.shape[2], HEAD_DIM=q.shape[3],
            SCALE=scale, CAUSAL=causal, BLOCK_M=block_m, BLOCK_N=block_n,
            num_warps=4, num_stages=2,
        )
    return out
