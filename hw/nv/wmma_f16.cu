#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#ifndef TENSORREV_ENABLE_WMMA_AMPERE
#define TENSORREV_ENABLE_WMMA_AMPERE 0
#endif

#if TENSORREV_ENABLE_WMMA_AMPERE
torch::Tensor mma_bf16(torch::Tensor A, torch::Tensor B_col, torch::Tensor C);
torch::Tensor mma_tf32(torch::Tensor A, torch::Tensor B_col, torch::Tensor C);
#endif

namespace {

__global__ void wmma_m16n16k16_fp16_fp32_kernel(
    const half* __restrict__ A,        // row-major 16x16, lda=16
    const half* __restrict__ B_col,    // col-major 16x16, ldb=16
    const float* __restrict__ C,       // row-major 16x16
    float* __restrict__ D              // row-major 16x16, ldc=16
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    constexpr int lda = 16;
    constexpr int ldb = 16;
    constexpr int ldc = 16;

    wmma::load_matrix_sync(a_frag, A, lda);
    wmma::load_matrix_sync(b_frag, B_col, ldb);
    wmma::load_matrix_sync(acc_frag, C, ldc, wmma::mem_row_major);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    wmma::store_matrix_sync(D, acc_frag, ldc, wmma::mem_row_major);
}

}  // namespace

torch::Tensor mma_f16(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
    auto D = torch::empty_like(C);

    dim3 block(32, 1, 1);
    dim3 grid(1, 1, 1);

    wmma_m16n16k16_fp16_fp32_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_col.data_ptr<at::Half>()),
        C.data_ptr<float>(),
        D.data_ptr<float>()
    );

    return D;
}

torch::Tensor mma_f16bf16tf32(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
    switch (A.scalar_type()) {
        case torch::kFloat16:
            return mma_f16(A, B_col, C);
#if TENSORREV_ENABLE_WMMA_AMPERE
        case torch::kBFloat16:
            return mma_bf16(A, B_col, C);
        case torch::kFloat32:
            return mma_tf32(A, B_col, C);
#endif
        default:
#if TENSORREV_ENABLE_WMMA_AMPERE
            TORCH_CHECK(false, "Unsupported dtype");
#else
            TORCH_CHECK(
                false,
                "Unsupported dtype for this WMMA build. "
                "Volta/Turing builds support only torch.float16; "
                "build for sm80+ to enable torch.bfloat16 and torch.float32 (TF32)."
            );
#endif
    }

    return torch::Tensor();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_f16bf16tf32", &mma_f16bf16tf32, "GEMM (FP16, plus TF32/BF16 on sm80+)");
}
