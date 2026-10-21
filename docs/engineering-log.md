# Preparation record

This records the actual engineering phases of the initial version. It is not a
reconstructed commit history.

1. **Inspect the technical environment.** An existing Python 3.12.14 interpreter
   was available on Windows 10 build 19045. The GPU inventory identified a
   GeForce GT 730 with driver 388.13 and 4096 MiB. PyTorch and Triton were absent.
   No drivers or development tools were installed during preparation.
2. **Set the scope.** The initial GPU and OS do not satisfy the intended
   Linux/NVIDIA target. The package therefore has a dependency-free executable
   model plus separately gated PyTorch/Triton paths. The tensor contract fixes
   contiguous layout, supported head dimensions, and upper-left causal
   alignment. Other features are listed as unsupported.
3. **Implement and review.** Implemented tiled online normalization, explicit
   dispatch, a framework oracle, tail masks and guarded experiment commands.
   Review tightened benchmark parameter checks before allocation and corrected
   environment eligibility to use the current CUDA device on mixed-GPU systems.
   Benchmark metadata records repeated-buffer cache behavior and sequential
   provider order. These are source-level changes, not measured optimizations.
4. **Validate what the machine can execute.** The CPU model was checked against
   a separate 70-digit Decimal calculation, algebraic properties, rectangular
   masks and failure cases. CLI/helper tests and package build checks ran.
   Optional framework/GPU tests retain explicit skip reasons. Exact counts are
   in [validation.md](validation.md) and the sanitized JSON evidence.
5. **Prepare the source snapshot.** Added English/Chinese documentation,
   environment pins and CI definitions. A source archive was generated with
   CRC and SHA256 checks. Generated build metadata and caches are excluded.
   Publication staging removes machine-local paths from environment evidence.

The upstream CUTLASS, AITER and Triton sources were read as engineering
references for contract checks, irregular-shape coverage and benchmark scope.
Their hardware results are not results for this project:

- [CUTLASS FMHA](https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/77_blackwell_fmha.cu)
- [AITER MHA tests](https://github.com/ROCm/aiter/blob/main/op_tests/test_mha.py)
- [Triton attention](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)

No GPU kernel compilation, device correctness run, latency measurement or
profiler session was possible on the initial machine. The next technical gate
is a real supported-GPU run.
