import importlib
import os
from pathlib import Path
from types import ModuleType

import torch


SUPPORTED_BACKENDS = {"auto", "nv", "mx"}
MX_DEVICE_KEYWORDS = ("metax", "muxi", "mxc")

_MX_EXTENSION_CACHE: dict[str, ModuleType] = {}


def _normalize_backend(raw_backend: str) -> str:
    backend = raw_backend.strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise RuntimeError(
            f"Unsupported TensorRev backend: {raw_backend}. "
            f"Supported backends: {sorted(SUPPORTED_BACKENDS)}"
        )
    return backend


def resolve_backend(device: int | None = None) -> str:
    configured_backend = _normalize_backend(os.environ.get("TENSORREV_BACKEND", "auto"))
    if configured_backend != "auto":
        return configured_backend

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if device is None:
        device = torch.cuda.current_device()

    device_name = torch.cuda.get_device_name(device).lower()
    if "nvidia" in device_name:
        return "nv"
    if any(keyword in device_name for keyword in MX_DEVICE_KEYWORDS):
        return "mx"

    # MX is currently the only non-NVIDIA prototype backend.
    return "mx"


def _import_first(candidates: tuple[str, ...]) -> ModuleType | None:
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def _load_mx_extension(module_name: str, source_name: str) -> ModuleType:
    cached = _MX_EXTENSION_CACHE.get(module_name)
    if cached is not None:
        return cached

    module = _import_first(
        (
            f"hw.{module_name}",
            f"hw.mx.{module_name}",
            module_name,
        )
    )
    if module is not None:
        _MX_EXTENSION_CACHE[module_name] = module
        return module

    source = Path(__file__).resolve().parents[1] / "hw" / "mx" / source_name
    if not source.is_file():
        raise RuntimeError(
            f"MX backend source not found: {source}. "
            "Apply the MX backend patch before selecting TENSORREV_BACKEND=mx."
        )

    from torch.utils.cpp_extension import load

    module = load(
        name=module_name,
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    _MX_EXTENSION_CACHE[module_name] = module
    return module


def _load_nv_wmma() -> ModuleType:
    module = _import_first(("hw.nv_wmma", "nv_wmma"))
    if module is None:
        raise RuntimeError("NV WMMA extension is not available. Build it with `make`.")
    return module


def _load_nv_mma() -> ModuleType:
    module = _import_first(("hw.nv_mma", "nv_mma"))
    if module is None:
        raise RuntimeError("NV F8 extension is not available. Build it with `make f8`.")
    return module


def get_mma_function(backend: str, dtype: torch.dtype):
    backend = _normalize_backend(backend)
    if backend == "auto":
        backend = resolve_backend()

    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        if backend == "nv":
            return _load_nv_wmma().mma_f16bf16tf32
        if backend == "mx":
            return _load_mx_extension("mx_wmma", "mx_wmma_f16bf16tf32.cu").mma_f16bf16tf32

    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if backend == "nv":
            return _load_nv_mma().cutlass_gemm_f8
        if backend == "mx":
            return _load_mx_extension("mx_mma", "mx_mma_f8.cu").mma_f8

    raise ValueError(f"Unsupported data type {dtype} for MMA operation on backend {backend}.")
