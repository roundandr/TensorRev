from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
COMMON_NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    # "-gencode=arch=compute_110,code=sm_110",
]

setup(
    name="mx_gemm_ext",
    ext_modules=[
        CUDAExtension(
            name="mx_wmma",
            sources=[str(ROOT / "mx_wmma_f16bf16tf32.cu")],
            extra_compile_args={"nvcc": COMMON_NVCC_FLAGS},
        ),
        CUDAExtension(
            name="mx_mma",
            sources=[str(ROOT / "mx_mma_f8.cu")],
            extra_compile_args={"nvcc": COMMON_NVCC_FLAGS},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
