# Reproduce the checks

Run commands from the extracted project directory. Python 3.10–3.12 is the baseline range. Commands below describe future reproducible runs; the checks actually performed on the initial machine are recorded separately in [validation.md](validation.md).

## 1. Standard-library checks

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m attention_kernel_lab doctor --json
python -m attention_kernel_lab demo --nq 7 --nk 11 --dim 4 --block-size 3 --causal
python -m unittest discover -s tests -v
```

Linux:

```bash
export PYTHONPATH=src
python -m attention_kernel_lab doctor --json
python -m attention_kernel_lab demo --nq 7 --nk 11 --dim 4 --block-size 3 --causal
python -m unittest discover -s tests -v
```

Use the path to an existing interpreter if `python` resolves to a nonfunctional Windows Store alias. No dependency installation is needed for this stage. Framework and GPU skips must remain visible in the test report.

## 2. Optional framework CPU environment

Create an isolated virtual environment; activate it before installing packages. On Windows its interpreter is `.venv\Scripts\python.exe`; on Linux it is `.venv/bin/python`.

```bash
python -m venv .venv
# Activate the environment for your shell, or use its interpreter directly.
python -m pip install -r requirements/cpu.txt
python -m pip install -e .
python -m attention_kernel_lab check --device cpu --backend reference --dtype fp32 --nq 33 --nk 65 --dim 32 --causal
python -m unittest discover -s tests -v
```

`requirements/cpu.txt` uses the official PyTorch CPU wheel index and pins `torch==2.5.1+cpu`. This is an optional download, not a requirement for the dependency-free model. Standard `unittest` runs all suites; pytest is available only as an optional developer extra.

## 3. Separate Linux GPU environment

Use an existing Linux x86-64 machine with an appropriate NVIDIA GPU and driver. The intended minimum capability is 8.0; a compatible framework wheel and driver are also required. Do not run this installation recipe on the initial GT 730 machine. No driver, WSL, or CUDA toolkit installation is performed by these instructions.

The historical baseline is Python 3.10–3.12, PyTorch 2.5.1 with CUDA 12.4 wheels, and Triton 3.1.0. [PyTorch's versioned installation page](https://pytorch.org/get-started/previous-versions/#v2-5-1) lists the CUDA 12.4 build; [its v2.5.1 Triton pin](https://github.com/pytorch/pytorch/blob/v2.5.1/.ci/docker/triton_version.txt) specifies 3.1.0. These pins are source-verified, not locally GPU-validated. They do not establish support for Blackwell or every newer GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/gpu-cu124.txt
python -m pip install -e .
python -m attention_kernel_lab doctor --json --output results/environment.json
python -m attention_kernel_lab check --device cuda --backend triton --dtype fp16 --nq 33 --nk 65 --dim 32 --causal
python -m unittest discover -s tests -v
```

Read the GPU suite output. If GPU tests skip, this run has not validated the GPU path. An eligible hardware/compiler failure is a test failure; do not replace it with a silent reference fallback.

## 4. Save measurements after correctness passes

```bash
python -m attention_kernel_lab bench --batch 1 --heads 4 --nq 512 --nk 512 --dim 64 --dtype fp16 --warmup 10 --samples 30 --iterations 10 --output results/fp16-d64.json
python -m attention_kernel_lab bench --batch 1 --heads 4 --nq 513 --nk 1025 --dim 64 --dtype bf16 --causal --warmup 10 --samples 30 --iterations 10 --output results/bf16-causal-tail.json
python -m attention_kernel_lab bench --dim 64 --block-m 16 --block-n 32 --include-eager --include-math --output results/tile-16x32.json
python -m pip freeze > results/pip-freeze.txt
```

Use `bench --help` for all parameters. The correctness gate compares providers with a dense FP32 reference before timing. CUDA matmul TF32 must be disabled for that oracle; the benchmark reports an error if it is enabled, leaving caller settings unchanged. Dense score elements are capped at 16,777,216 and total Q/K/V input elements at 134,217,728 to bound the harness; larger experiments need a separately designed validation strategy.

JSON includes input configuration, environment/device versions, numerical errors, tolerance settings, individual sample latencies, median/p20/p80, a logical FLOP-rate estimate, and the measured ratio to `sdpa_auto`. The latter lets PyTorch select its own backend and may already use fused attention. `--include-eager` adds the explicit framework reference; `--include-math` requests the framework math SDPA baseline. Neither replaces `sdpa_auto` as the main comparison.

CUDA-event timing excludes initial compilation, warmup, input construction/transfers, and backward. Timed calls allocate their output. The same input buffers are reused without cache rotation or flush; results describe that repeated-buffer workload, not cold DRAM traffic. Providers are measured sequentially. Save idle-device conditions and note any contention or thermal changes when interpreting results. Logical TFLOPS is an accounting estimate, not a hardware utilization measurement.

## 5. CI and evidence

The CI file defines Windows/Linux Python 3.10/3.11/3.12 standard-library and optional torch-CPU jobs. It has not been executed on GitHub during local preparation. GPU CI is opt-in through `workflow_dispatch` and requires an owner-provisioned runner labelled `self-hosted, linux, x64, nvidia-sm80`. It checks actual SM80 hardware and treats skipped GPU tests as a failed validation job. No runner is created by this repository.

For each actual GPU experiment preserve the command, complete test output, doctor JSON, benchmark JSON, and dependency versions. Report failures and skips alongside passes. Update [validation.md](validation.md) only from those records.
