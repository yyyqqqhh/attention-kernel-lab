# Forward attention design

## Operation and invariants

For each batch and head, let `Q` have shape `[M,D]` and `K,V` have shape `[N,D]`. The output is

```text
S[i,j] = scale * dot(Q[i], K[j])
P[i,:] = softmax(S[i,:])
O[i,:] = sum_j P[i,j] * V[j,:]
```

Causal mode assigns negative infinity to scores with `j > i`. Alignment is upper-left for both square and rectangular inputs. For example, `M=2,N=4` allows keys `{0}` for query 0 and `{0,1}` for query 1. This is not a chunk-offset or right-aligned KV-cache rule.

All dimensions are positive, so every real row has at least one visible key in the supported masking modes. Inputs must be finite and within a range where FP32 dot-product accumulation remains finite. The tensor path does not scan every element for NaN or infinity: that would add a synchronization and data pass to every launch.

## Incremental normalization

The kernel never needs the complete `[M,N]` score or probability tensor in device memory. For a query tile it retains, per row, running maximum `m`, denominator `l`, and unnormalized weighted output vector `a`. Start with `m=-inf`, `l=0`, and `a=0`.

For each key/value tile, calculate scores `s`, apply masks, and update:

```text
m_new = max(m, rowmax(s))
alpha = exp(m - m_new)
p     = exp(s - m_new)
l_new = alpha*l + rowsum(p)
a_new = alpha*a + p @ V_tile
```

Finish with `O=a/l`. Rescaling the previous partial sums when the maximum changes preserves their interpretation under the new normalization. This is an exact reformulation in real arithmetic; floating-point reductions and the low-precision matrix products introduce rounding differences.

The standard-library implementation exposes the recurrence at a small scale. It serves as an executable model, not as a model of GPU instruction scheduling or GPU rounding. The PyTorch reference materializes the score matrix to provide a simple independent computation path.

## Mapping to Triton

One program instance owns a query tile within one batch/head. Query elements are reused while the program advances over key/value tiles. FP32 holds maxima, denominators, and output accumulation. Matrix products use the supported input types; probabilities must be converted to a compatible low-precision operand for the value product.

Boundary masks protect partial query and key tiles. Masked scores do not contribute to normalization, and invalid query lanes must not leak a NaN into a real output lane. Output stores are masked. A small fixed tile interface keeps specialization explicit and avoids concealing an unmeasured autotuning policy.

The mathematical work remains quadratic in the two sequence lengths: approximately `4*B*H*M*N*D` matrix-product floating-point operations before masking savings and softmax overhead. The change is how intermediate data is retained. Per-program live state depends on tile sizes, including score storage proportional to `BLOCK_M*BLOCK_N` and accumulator storage proportional to `BLOCK_M*D`; it does not allocate the full score matrix in global memory. Compiler spilling can still create memory traffic. GPU profiling is needed to establish register use and actual traffic.

## Backend behavior

The package's standard-library path imports without torch or Triton. Framework imports are deferred until their APIs are used. `reference` selects the framework implementation. `triton` demands that all GPU preconditions are met. `auto` uses reference when those preconditions are absent and surfaces real compilation or execution failures when the GPU path is eligible. An explicit GPU request cannot quietly produce a CPU/reference benchmark.

The GPU wrapper rejects tensors requiring gradients. There is no custom backward implementation and no claim of training support. Layout normalization is an explicit caller responsibility; silently making contiguous copies would hide allocation and copy costs.

## Measurement questions

| Hypothesis | Evidence needed |
| --- | --- |
| Larger query tiles reduce repeated key/value reads | Latency and profiler traffic counters at fixed shape and software stack |
| Larger tiles lose to register pressure at large `D` | Register/spill statistics, achieved occupancy, and latency |
| Causal iteration can skip entirely future tiles | Equal-output checks for rectangular/tail cases, then measured work reduction |
| The framework wins on some shapes | Direct `sdpa_auto` measurements with backend and environment recorded |

These are proposed experiments, not results. No architecture-specific performance conclusion follows from the CPU model.

## References

The recurrence follows the public [online normalizer](https://arxiv.org/abs/1805.02867) idea. The IO-aware attention organization is described in [FlashAttention](https://arxiv.org/abs/2205.14135). [Triton's attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) is an external implementation reference; its advanced scheduling paths and published benchmark numbers are not attributes of this project.
