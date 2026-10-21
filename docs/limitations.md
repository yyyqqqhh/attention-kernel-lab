# Supported scope and failure boundaries

This project has not yet run its Triton kernel on a supported GPU. Source review and CPU algorithm tests cannot establish device compilation, instruction behavior, numerical tolerance, or latency. See [validation.md](validation.md) for the executed checks.

## Interface boundaries

The Triton path accepts contiguous rank-four `q[B,H,M,D]`, `k[B,H,N,D]`, and `v[B,H,N,D]` tensors with matching dtype/device/batch/head/head-dimension. Dimensions must be positive; `D` is 16, 32, 64, or 128; dtype is fp16 or bf16. Query and key lengths may differ and may have partial final tiles. Scale is finite and positive. `block_m` and `block_n` are each 16, 32, or 64. Each input must contain fewer than `2**31` elements, and `B*H` must not exceed 65535; these bounds keep pointer arithmetic and the batch/head launch grid within the implemented contract.

Unsupported features include backward/autograd, dropout, arbitrary additive/boolean masks, padding masks, GQA/MQA, variable-length packed batches, ragged tensors, KV-cache offsets, rotary embedding fusion, quantized/FP8/FP32 GPU inputs, noncontiguous layouts, and differing key/value head dimensions. The implementation does not claim compatibility with `torch.compile`, CUDA graph capture, distributed execution, or export runtimes.

Upper-left causal alignment is a deliberate semantic choice. For an incremental decoder whose query indices start later in a KV cache, this interface does not express the required offset. Do not use `causal=True` as an implicit cache-offset mask.

## Platform boundaries

The intended kernel target is Linux with NVIDIA CUDA compute capability at least 8.0 and a compatible driver/compiler/framework combination. ROCm is rejected. Windows can run the CPU code and optional PyTorch CPU tests, but native Windows Triton variants are outside this project. No driver, CUDA toolkit, WSL installation, or system setting is changed by the project.

The supplied CUDA 12.4 / PyTorch 2.5.1 / Triton 3.1.0 environment is a reproducible starting point for an appropriate older target, not a promise of support for every later architecture. Blackwell, MI300, AMD, and other hardware are unvalidated. A minimum-capability check does not guarantee that the installed wheel contains code for a particular device.

The initial GT 730 / driver 388.13 machine does not satisfy the target. Attempting to install a large modern CUDA stack there would not turn CPU checks into relevant GPU evidence.

## Numerical boundaries

Tensor inputs are assumed finite and not so large that score accumulation overflows FP32. FP16/BF16 inputs and the probability operand used by the value product introduce rounding. FP32 accumulation reduces error; it does not make the result bitwise equal to the reference. Tile partitioning can change reduction order.

Accuracy checks use dtype-dependent absolute and relative tolerances and report measured errors. CUDA FP32 oracles must disable TF32; the benchmark refuses an enabled TF32 policy rather than changing global settings in a caller's process. The general reference API also leaves that policy unchanged. Relative error near a zero reference value is ill-conditioned, so absolute error and a stabilized relative measure must be read together. Passing a small seeded random case does not validate all values. Tail shapes, rectangular masks, scaled inputs, and adversarial score ranges matter.

There is no arbitrary mask that can fully mask a real row in the supported interface. If that feature is added, its output semantics and the `-inf - -inf` transition must be specified and tested before use.

## Performance and experiment boundaries

There is no measured speedup, TFLOPS, peak-memory claim, or hardware ranking. The benchmark's `sdpa_auto` baseline can already use a framework fused implementation. A comparison against eager attention alone would not establish superiority over framework SDPA.

The explicit reference creates a quadratic score tensor and can run out of memory long before the tiled path. Start with the bounded defaults and increase shapes deliberately. Timing reuses the same Q/K/V buffers without cache rotation or flush, allocates outputs per call, and measures providers sequentially. It measures a steady-state forward call after warmup; it does not measure model throughput, training, compilation latency, cache-cold traffic, power, or multi-tenant behavior. Device contention and thermal state can affect results.

Top-level package versions are pinned for the example environments, but transitive dependencies, driver version, operating system, and hardware are not fully locked. Save the emitted environment metadata and `pip freeze` for each actual experiment.

## Publication boundaries

No GPU benchmark results are included. A distribution license has not been selected; see [NOTICE.md](../NOTICE.md). The project includes no private employer code, asserted deployment history, invented education dates, or invented performance achievements.
