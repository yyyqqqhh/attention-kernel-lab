"""Small, read-only runtime inventory. Importing this module needs no extras."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(*, probe_torch: bool = True) -> dict:
    """Inspect available tools; never install drivers or modify the machine."""
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": {name: _version(name) for name in ("torch", "triton", "pytest")},
        "tools": {name: shutil.which(name) for name in ("nvidia-smi", "nvcc", "git")},
        "cuda_available": None,
        "gpu_target": "Linux, NVIDIA CUDA, compute capability >= 8.0; unverified until tested",
    }
    nvidia_smi = result["tools"]["nvidia-smi"]
    if not nvidia_smi and platform.system() == "Windows":
        for candidate in (
            Path("C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"),
            Path("C:/Windows/System32/nvidia-smi.exe"),
        ):
            if candidate.is_file():
                nvidia_smi = str(candidate)
                result["tools"]["nvidia-smi"] = nvidia_smi
                break
    if nvidia_smi:
        try:
            process = subprocess.run(
                [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            result["nvidia_smi"] = {
                "returncode": process.returncode,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["nvidia_smi_error"] = str(exc)
    if probe_torch and importlib.util.find_spec("torch") is not None:
        try:
            import torch

            result["torch_runtime"] = {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "hip_version": torch.version.hip,
                "cuda_available": torch.cuda.is_available(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            }
            result["cuda_available"] = torch.cuda.is_available()
            if result["cuda_available"]:
                result["current_device"] = torch.cuda.current_device()
                result["devices"] = [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i)),
                        "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    }
                    for i in range(torch.cuda.device_count())
                ]
        except Exception as exc:  # Diagnostics must remain useful for broken optional installs.
            result["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    result["gpu_preflight_ready"] = bool(
        result["system"] == "Linux"
        and result.get("cuda_available")
        and not result.get("torch_runtime", {}).get("hip_version")
        and result["packages"]["triton"]
        and any(device["index"] == result.get("current_device") and device["capability"][0] >= 8 for device in result.get("devices", []))
    )
    result["validation_note"] = (
        "Inventory is a preflight check, not evidence that a Triton kernel compiled or ran."
    )
    return result
