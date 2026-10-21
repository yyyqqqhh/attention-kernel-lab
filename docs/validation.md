# Local validation

Prepared on 2026-10-21. These results describe the local source snapshot only.
These are the pre-publication local checks. Remote CI and supported-GPU results are not included in this snapshot.

| Check | Actual result |
| --- | --- |
| Python | Existing CPython 3.12.14, Windows 10 build 19045 |
| GPU inventory | GeForce GT 730, 4096 MiB, NVIDIA driver 388.13 |
| Optional packages | PyTorch, Triton and pytest absent |
| Syntax check | 15 Python files parsed successfully with Python AST; not Triton compilation |
| Full unittest discovery | 33 methods: 18 passed, 15 explicitly skipped, 0 failures/errors |
| Independent oracle | Online and dense CPU results checked against 70-digit Decimal arithmetic |
| CPU demo | Causal Nq=9, Nk=13, D=4, tile=3; maximum absolute difference 2.220446049250313e-16 |
| Packaging | Local wheel build and isolated target-directory install succeeded with no dependency downloads |
| GPU compilation/correctness | Not run |
| GPU benchmark | Not run; no latency, speedup or throughput data |
| Remote CI | Configured, not run |

The 15 skipped methods comprise 10 optional PyTorch methods and 5 GPU methods.
GPU methods contain further dtype, dimension, tile and shape subcases; none was
executed. A skipped method is not a passing GPU check.

Executed source commands, using the available Python interpreter:

```text
python tools/validate.py
python -m attention_kernel_lab doctor --output results/environment.json
python -m attention_kernel_lab demo --causal --nq 9 --nk 13 --block-size 3 --output results/cpu-demo.json
python -m pip install . --no-deps --no-build-isolation --no-index --target <dedicated-validation-directory>
```

The source commands use `PYTHONPATH=src`; `tools/validate.py` sets its own source
path. Missing-dependency `check` and `bench` requests were also checked: each
returns exit code 2 with a clear explanation and produces no fake result file.

Machine-readable evidence:

- [Full validation log and skip reasons](../results/validation.json)
- [Read-only environment inventory](../results/environment.json)
- [CPU demonstration](../results/cpu-demo.json)
- [Actual implementation phases](engineering-log.md)

The CPU model uses Python double precision. It verifies the recurrence and mask
semantics, not GPU rounding, register allocation, compiler lowering, race
behavior or performance. Static review found no source-level correctness
blocker, but only real device execution can establish the GPU path.

The target GPU gate remains pending because the initial machine lacks the required
hardware/runtime. No system changes or dependency installs were needed for these
local checks. Environment evidence is sanitized to omit local executable paths.
