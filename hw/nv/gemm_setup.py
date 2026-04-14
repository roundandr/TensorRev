import os
import re
import sys
from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
WMMA_MINIMUM_ARCH = 80
F8_MINIMUM_ARCH = 90


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value}")


def _normalize_cuda_arch(raw_arch: str) -> str:
    arch = raw_arch.strip().lower()
    if not arch:
        raise ValueError("CUDA arch must not be empty")
    if arch.startswith("sm_"):
        arch = arch[3:]
    if arch.startswith("compute_"):
        arch = arch[8:]
    return arch


def _extract_arch_number(cuda_arch: str) -> int:
    match = re.match(r"^(\d+)", cuda_arch)
    if match is None:
        raise ValueError(f"Invalid CUDA arch value: {cuda_arch}")
    return int(match.group(1))


def _resolve_cuda_arch() -> str:
    configured_arch = os.environ.get("TENSORREV_CUDA_ARCH", "auto")
    if configured_arch.strip().lower() != "auto":
        return _normalize_cuda_arch(configured_arch)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA arch detection requires an available CUDA device. "
            "Set TENSORREV_CUDA_ARCH (or CUDA_ARCH in TensorRev/build_config.mk) explicitly."
        )

    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}"


def _resolve_cutlass_dir() -> Path:
    raw_path = os.environ.get("TENSORREV_CUTLASS_DIR", "").strip()
    if not raw_path:
        raise RuntimeError(
            "CUTLASS_DIR is not configured. "
            "Set it in TensorRev/build_config.mk before building F8."
        )

    cutlass_dir = Path(raw_path).expanduser()
    if not (cutlass_dir / "include").is_dir():
        raise FileNotFoundError(
            f"CUTLASS include directory not found: {cutlass_dir / 'include'}"
        )
    if not (cutlass_dir / "tools" / "util" / "include").is_dir():
        raise FileNotFoundError(
            f"CUTLASS util include directory not found: {cutlass_dir / 'tools' / 'util' / 'include'}"
        )
    return cutlass_dir


def _assert_f8_arch_supported(cuda_arch: str) -> None:
    arch_number = _extract_arch_number(cuda_arch)
    if arch_number < F8_MINIMUM_ARCH:
        raise RuntimeError(
            "cutlass_gemm_f8 is only enabled for CUDA arch >= 90. "
            f"Resolved arch: {cuda_arch}. "
            "This check matches the current kernel implementation, which supports cutlass::arch::Sm90 and Sm100."
        )


def _assert_wmma_arch_supported(cuda_arch: str) -> None:
    arch_number = _extract_arch_number(cuda_arch)
    if arch_number < WMMA_MINIMUM_ARCH:
        raise RuntimeError(
            "wmma_f16bf16tf32 requires CUDA arch >= 80. "
            f"Resolved arch: {cuda_arch}."
        )


def _check_wmma_support() -> str:
    cuda_arch = _resolve_cuda_arch()
    _assert_wmma_arch_supported(cuda_arch)
    return cuda_arch


def _check_f8_support() -> str:
    cuda_arch = _resolve_cuda_arch()
    _assert_wmma_arch_supported(cuda_arch)
    _assert_f8_arch_supported(cuda_arch)
    return _resolve_f8_cuda_arch(cuda_arch)


def _handle_custom_cli() -> None:
    if "--check-wmma-support" in sys.argv:
        sys.argv.remove("--check-wmma-support")
        cuda_arch = _check_wmma_support()
        print(f"WMMA build is supported for CUDA arch {cuda_arch}.")
        raise SystemExit(0)

    if "--check-f8-support" not in sys.argv:
        return

    sys.argv.remove("--check-f8-support")
    cuda_arch = _check_f8_support()
    print(f"F8 build is supported for CUDA arch {cuda_arch}.")
    raise SystemExit(0)


def _resolve_f8_kernel_arch(cuda_arch: str) -> int:
    arch_number = _extract_arch_number(cuda_arch)
    if arch_number >= 100:
        return 100
    return 90


def _resolve_f8_cuda_arch(cuda_arch: str) -> str:
    if cuda_arch.endswith("a"):
        return cuda_arch
    return f"{cuda_arch}a"


def _build_common_nvcc_flags() -> list[str]:
    cuda_arch = _resolve_cuda_arch()
    _assert_wmma_arch_supported(cuda_arch)
    build_cutlass_gemm_f8 = _parse_bool_env("TENSORREV_BUILD_CUTLASS_GEMM_F8", False)
    nvcc_arch = cuda_arch
    if build_cutlass_gemm_f8:
        _assert_f8_arch_supported(cuda_arch)
        nvcc_arch = _resolve_f8_cuda_arch(cuda_arch)

    flags = [
        "-O2",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{nvcc_arch},code=sm_{nvcc_arch}",
    ]

    if build_cutlass_gemm_f8:
        cutlass_dir = _resolve_cutlass_dir()
        flags.extend(
            [
                f"-I{cutlass_dir / 'include'}",
                f"-I{cutlass_dir / 'tools' / 'util' / 'include'}",
            ]
        )

    return flags


_handle_custom_cli()

COMMON_NVCC_FLAGS = _build_common_nvcc_flags()
BUILD_CUTLASS_GEMM_F8 = _parse_bool_env("TENSORREV_BUILD_CUTLASS_GEMM_F8", False)
CUDA_ARCH = _resolve_cuda_arch()

ext_modules = [
    CUDAExtension(
        name="nv_wmma",
        sources=[str(ROOT / "wmma_f16bf16tf32.cu")],
        extra_compile_args={
            "cxx": ["-O2", "-std=c++17"],
            "nvcc": COMMON_NVCC_FLAGS,
        },
    ),
]

if BUILD_CUTLASS_GEMM_F8:
    f8_nvcc_flags = COMMON_NVCC_FLAGS + [
        f"-DTENSORREV_F8_ARCH_NUMBER={_resolve_f8_kernel_arch(CUDA_ARCH)}",
    ]
    ext_modules.append(
        CUDAExtension(
            name="nv_mma",
            sources=[str(ROOT / "cutlass_gemm_f8.cu")],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],
                "nvcc": f8_nvcc_flags,
            },
        )
    )

setup(
    name="nv_gemm_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
