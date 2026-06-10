import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent
WMMA_MINIMUM_ARCH = 70
WMMA_AMPERE_MINIMUM_ARCH = 80
F8_MINIMUM_ARCH = 89
ADA_F8_MINIMUM_CUDA = (12, 4)
SM120_F8_MINIMUM_CUDA = (12, 8)
SM121_F8_MINIMUM_CUDA = (12, 9)


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


def _load_torch():
    import torch

    return torch


def _normalize_cuda_arch(raw_arch: str) -> str:
    arch = raw_arch.strip().lower()
    if not arch:
        raise ValueError("CUDA arch must not be empty")
    if arch.startswith("sm_"):
        arch = arch[3:]
    if arch.startswith("compute_"):
        arch = arch[8:]
    if re.match(r"^\d+a?$", arch) is None:
        raise ValueError(f"Invalid CUDA arch value: {raw_arch}")
    if arch.endswith("a") and _extract_arch_number(arch) < 90:
        raise ValueError(f"Invalid CUDA arch value: {raw_arch}. The 'a' suffix is only valid for sm90+ targets.")
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

    torch = _load_torch()
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


def _find_nvcc() -> Path | None:
    cudacxx = os.environ.get("CUDACXX", "").strip()
    if cudacxx:
        return Path(cudacxx).expanduser()

    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_root = os.environ.get(env_name, "").strip()
        if cuda_root:
            candidate = Path(cuda_root).expanduser() / "bin" / "nvcc"
            if candidate.is_file():
                return candidate

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        return Path(nvcc)

    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        CUDA_HOME = None

    if CUDA_HOME:
        candidate = Path(CUDA_HOME) / "bin" / "nvcc"
        if candidate.is_file():
            return candidate

    return None


def _parse_nvcc_version(output: str) -> tuple[int, int] | None:
    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if match is None:
        match = re.search(r"\bV(\d+)\.(\d+)", output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _resolve_nvcc_version() -> tuple[int, int]:
    nvcc = _find_nvcc()
    if nvcc is None:
        raise RuntimeError(
            "sm89 FP8 builds require CUDA toolkit >= 12.4, but nvcc was not found. "
            "Set CUDACXX, CUDA_HOME, or CUDA_PATH to the CUDA toolkit used for this build."
        )

    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Failed to query nvcc version from {nvcc}: {exc}") from exc

    output = result.stdout + result.stderr
    version = _parse_nvcc_version(output)
    if version is None:
        raise RuntimeError(f"Failed to parse nvcc version from {nvcc} output:\n{output}")
    return version


def _assert_minimum_nvcc_version(minimum: tuple[int, int], reason: str) -> None:
    version = _resolve_nvcc_version()
    if version < minimum:
        minimum_str = ".".join(str(part) for part in minimum)
        detected = ".".join(str(part) for part in version)
        raise RuntimeError(
            f"{reason} require CUDA toolkit >= {minimum_str}. "
            f"Detected nvcc version: {detected}."
        )


def _assert_f8_toolkit_supported(cuda_arch: str) -> None:
    arch_number = _extract_arch_number(cuda_arch)
    kernel_arch = _resolve_f8_kernel_arch(cuda_arch)
    if kernel_arch == 89:
        _assert_minimum_nvcc_version(
            ADA_F8_MINIMUM_CUDA,
            "sm89 FP8 builds",
        )
    if arch_number >= 121:
        _assert_minimum_nvcc_version(
            SM121_F8_MINIMUM_CUDA,
            f"sm{arch_number} FP8 builds",
        )
    elif kernel_arch >= 120:
        _assert_minimum_nvcc_version(
            SM120_F8_MINIMUM_CUDA,
            "sm120 FP8 builds",
        )


def _assert_f8_arch_supported(cuda_arch: str) -> None:
    arch_number = _extract_arch_number(cuda_arch)
    if arch_number < F8_MINIMUM_ARCH:
        raise RuntimeError(
            "cutlass_gemm_f8 is only enabled for CUDA arch >= 89. "
            f"Resolved arch: {cuda_arch}. "
            "This check matches the current kernel implementation, which supports cutlass::arch::Sm89, Sm90, Sm100, and Sm120."
        )


def _assert_wmma_arch_supported(cuda_arch: str) -> None:
    arch_number = _extract_arch_number(cuda_arch)
    if arch_number < WMMA_MINIMUM_ARCH:
        raise RuntimeError(
            "nv_wmma requires CUDA arch >= 70. "
            f"Resolved arch: {cuda_arch}. "
            "sm70/sm75 builds include only FP16 WMMA; sm80+ builds add BF16 and TF32."
        )


def _supports_wmma_ampere(cuda_arch: str) -> bool:
    return _extract_arch_number(cuda_arch) >= WMMA_AMPERE_MINIMUM_ARCH


def _describe_wmma_build(cuda_arch: str) -> str:
    if _supports_wmma_ampere(cuda_arch):
        return "FP16/BF16/TF32"
    return "FP16-only"


def _check_wmma_support() -> str:
    cuda_arch = _resolve_cuda_arch()
    _assert_wmma_arch_supported(cuda_arch)
    return cuda_arch


def _check_f8_support() -> str:
    cuda_arch = _resolve_cuda_arch()
    _assert_wmma_arch_supported(cuda_arch)
    _assert_f8_arch_supported(cuda_arch)
    _assert_f8_toolkit_supported(cuda_arch)
    return _resolve_f8_cuda_arch(cuda_arch)


def _handle_custom_cli() -> None:
    if "--check-wmma-support" in sys.argv:
        sys.argv.remove("--check-wmma-support")
        cuda_arch = _check_wmma_support()
        print(
            f"WMMA build is supported for CUDA arch {cuda_arch} "
            f"({_describe_wmma_build(cuda_arch)})."
        )
        raise SystemExit(0)

    if "--check-f8-support" not in sys.argv:
        return

    sys.argv.remove("--check-f8-support")
    cuda_arch = _check_f8_support()
    print(f"F8 build is supported for CUDA arch {cuda_arch}.")
    raise SystemExit(0)


def _resolve_f8_kernel_arch(cuda_arch: str) -> int:
    arch_number = _extract_arch_number(cuda_arch)
    if arch_number >= 120:
        return 120
    if arch_number >= 100:
        return 100
    if arch_number >= 90:
        return 90
    return 89


def _resolve_f8_cuda_arch(cuda_arch: str) -> str:
    arch_number = _extract_arch_number(cuda_arch)
    if arch_number == 89:
        if cuda_arch.endswith("a"):
            raise ValueError("sm89 FP8 builds must target CUDA arch 89, not 89a.")
        return cuda_arch
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
        _assert_f8_toolkit_supported(cuda_arch)
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


def _build_wmma_sources(cuda_arch: str) -> list[str]:
    sources = [
        ROOT / "wmma_f16.cu",
    ]
    if _supports_wmma_ampere(cuda_arch):
        sources.append(ROOT / "wmma_bf16tf32.cu")
    return [str(source) for source in sources]


def _build_wmma_define(cuda_arch: str) -> str:
    enabled = int(_supports_wmma_ampere(cuda_arch))
    return f"-DTENSORREV_ENABLE_WMMA_AMPERE={enabled}"


_handle_custom_cli()

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

COMMON_NVCC_FLAGS = _build_common_nvcc_flags()
BUILD_CUTLASS_GEMM_F8 = _parse_bool_env("TENSORREV_BUILD_CUTLASS_GEMM_F8", False)
CUDA_ARCH = _resolve_cuda_arch()
WMMA_DEFINE = _build_wmma_define(CUDA_ARCH)

ext_modules = [
    CUDAExtension(
        name="nv_wmma",
        sources=_build_wmma_sources(CUDA_ARCH),
        extra_compile_args={
            "cxx": ["-O2", "-std=c++17", WMMA_DEFINE],
            "nvcc": COMMON_NVCC_FLAGS + [WMMA_DEFINE],
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
