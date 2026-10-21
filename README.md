# attention-kernel-lab

A small forward-attention project for studying online softmax, GPU tiling, and the cost of moving intermediate tensors through memory.

[中文说明](README.zh-CN.md) · [Design](docs/design.md) · [Reproduce](docs/reproduce.md) · [Limitations](docs/limitations.md) · [Validation record](docs/validation.md)

The repository includes a dependency-free CPU model of the algorithm, an explicit PyTorch reference, an actual Triton kernel, contract and numerical tests, and a benchmark CLI. It is an engineering experiment with a narrow interface. It is not a production attention library.

**Validation status:** the initial machine is Windows 10 with a GeForce GT 730 and driver 388.13. That machine cannot validate the intended GPU path. GPU compilation, GPU correctness, and performance measurements are pending. There are no measured speedup claims. The exact locally executed checks belong in [the validation record](docs/validation.md).

## What is implemented

| Component | Purpose |
| --- | --- |
| `online.py` | Dense and blocked online attention using only Python lists and the standard library; executable algorithm and masking checks without a GPU |
| `torch_ops.py` | FP32 intermediate PyTorch reference, argument validation, explicit backend selection |
| `triton_kernel.py` | Query tiles stream through key/value tiles while retaining row maxima, normalizers, and weighted sums |
| `benchmark.py` | Correctness gate, CUDA-event timing, warmup, repeated samples, environment and error metadata |
| `tests/` | CPU algorithm tests, optional framework tests, hardware-gated GPU tests |
| `.github/workflows/ci.yml` | CPU jobs and an opt-in GPU job on an owner-provisioned runner |

## Start without a GPU or framework

Use an existing Python 3.10–3.12 interpreter. The standard-library path does not install packages or change the system.

```powershell
# Windows PowerShell, from the extracted project directory
$env:PYTHONPATH = "src"
python -m attention_kernel_lab demo
python -m attention_kernel_lab doctor
python -m unittest discover -s tests -v
```

```bash
# Linux, from the extracted project directory
export PYTHONPATH=src
python -m attention_kernel_lab demo
python -m attention_kernel_lab doctor
python -m unittest discover -s tests -v
```

Framework and GPU tests report explicit skips when their dependencies or hardware are unavailable. A CPU-only passing run is not GPU validation.

To install the command-line entry point in an isolated environment:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -e .
attention-kernel-lab --help
```

## Tensor contract

```python
from attention_kernel_lab.torch_ops import attention

# q: [B, H, M, D], k/v: [B, H, N, D]
# Allocate contiguous CUDA fp16/bf16 tensors on a supported target first.
out = attention(q, k, v, causal=True, backend="triton")
```

| Property | Triton path |
| --- | --- |
| Operation | `softmax(scale * Q @ K.T) @ V`, forward inference only |
| Shape | Positive `B, H, M, N`; `D` in `{16, 32, 64, 128}`; equal batch, heads, and head dimensions |
| Layout | Contiguous rank-four tensors on one device |
| Dtype | Matching `float16` or `bfloat16`; FP32 softmax state and output accumulation |
| Output | `[B, H, M, D]`, same dtype/device as the input |
| Causal rule | Upper-left alignment: key `j` is visible to query `i` when `j <= i`, including rectangular shapes |
| Scale | Default `1 / sqrt(D)`; explicit scale must be finite and positive |
| Tails | Sequence lengths need not be tile multiples; boundary loads/stores are masked |
| Launch bounds | `block_m`/`block_n` each in `{16, 32, 64}`; `B*H <= 65535`; each input has fewer than `2**31` elements |
| Intended platform | Linux, NVIDIA CUDA compute capability at least 8.0, compatible framework/compiler/driver |

This platform check is an eligibility condition, not evidence that every later NVIDIA architecture works with the pinned environment. Blackwell and AMD/MI300 support are not claimed. CPU and framework references serve as oracles, with their own broader dtype/platform capabilities.

`backend="triton"` fails clearly if its contract is not met. `backend="auto"` may select the reference when the GPU path is ineligible; a compiler or launch failure on an eligible path is surfaced. Tensors requiring gradients are rejected by the Triton path. The [limitations](docs/limitations.md) describe unsupported features and numerical assumptions.

## Framework and GPU runs

CPU framework dependencies are optional:

```bash
python -m pip install -r requirements/cpu.txt
python -m pip install -e .
python -m unittest discover -s tests -v
```

For a separate Linux CUDA machine, [the reproduction guide](docs/reproduce.md) provides a PyTorch 2.5.1 / CUDA 12.4 / Triton 3.1.0 baseline. These are coherent historical top-level pins, not a fully locked environment or a locally GPU-tested stack. Do not install this GPU environment on the initial GT 730 machine.

```bash
python -m attention_kernel_lab check --help
python -m attention_kernel_lab bench --help
```

The benchmark compares this explicit Triton path with PyTorch SDPA under its default backend selection, labelled `sdpa_auto`. The framework may itself choose an optimized fused kernel. Correctness is checked before timing. Read the emitted metadata and error values before interpreting latency. Use measured JSON from the target machine; no sample performance results are committed.

## Next experiments

The useful next step is a supported GPU run of the complete correctness matrix, followed by fixed-shape tile sweeps. Larger tiles may reduce repeated key/value reads but increase register pressure and reduce occupancy. Causal work partitioning and head-dimension specialization are further hypotheses to measure. None of these hypotheses is a measured result.

## Sources and provenance

The project implements the public online-normalization and IO-aware attention ideas for inspection and experiments. It contains no company-internal source or asserted production deployment history.

- [FlashAttention paper](https://arxiv.org/abs/2205.14135) — tiling attention around memory traffic.
- [Online normalizer calculation](https://arxiv.org/abs/1805.02867) — incremental softmax normalization.
- [Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) — public compiler/kernel reference; tutorial performance numbers are not results for this project.
- [PyTorch SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — framework semantics and backend selection.
- [PyTorch historical installation instructions](https://pytorch.org/get-started/previous-versions/#v2-5-1), [PyTorch v2.5.1 Triton version](https://github.com/pytorch/pytorch/blob/v2.5.1/.ci/docker/triton_version.txt) — environment baseline.

See [NOTICE.md](NOTICE.md) for provenance and license status. No distribution license or license grant has been selected.
