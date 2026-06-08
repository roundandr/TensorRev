#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void wmma_m16n16k16_fp16_fp32_kernel(
    const half* __restrict__ A,        // row-major 16x16, lda=16
    const half* __restrict__ B_col,    // col-major 16x16, ldb=16 (but stored as row-major of B^T)
    const float* __restrict__ C,       // row-major 16x16,
    float* __restrict__ D              // row-major 16x16, ldc=16
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // Leading dimensions
    constexpr int lda = 16;
    constexpr int ldb = 16; // for col_major B
    constexpr int ldc = 16;
    wmma::load_matrix_sync(a_frag, A, lda);
    wmma::load_matrix_sync(b_frag, B_col, ldb);
    wmma::load_matrix_sync(acc_frag, C, ldc, wmma::mem_row_major);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    wmma::store_matrix_sync(D, acc_frag, ldc, wmma::mem_row_major);
}

__global__ void wmma_m16n16k8_tf32_fp32_kernel(
    const float* __restrict__ A,       // row-major 16x8, lda=8
    const float* __restrict__ B_col,   // col-major 8x16, ldb=16
    const float* __restrict__ C,       // row-major 16x16
    float* __restrict__ D              // row-major 16x16
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;

    constexpr int lda = 8;
    constexpr int ldb = 16;   // col-major B: logical shape is 8x16
    constexpr int ldc = 16;

    wmma::load_matrix_sync(a_frag, A, lda);
    wmma::load_matrix_sync(b_frag, B_col, ldb);
    wmma::load_matrix_sync(acc_frag, C, ldc, wmma::mem_row_major);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

    wmma::store_matrix_sync(D, acc_frag, ldc, wmma::mem_row_major);
}

__global__ void wmma_m16n16k16_bf16_fp32_kernel(
    const __nv_bfloat16* __restrict__ A,      // row-major 16x16, lda=16
    const __nv_bfloat16* __restrict__ B_col,  // col-major 16x16, ldb=16
    const float* __restrict__ C,              // row-major 16x16
    float* __restrict__ D                     // row-major 16x16
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
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

torch::Tensor mma_f16bf16tf32(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
    auto D = torch::empty_like(C);

    dim3 block(64, 1, 1);
    dim3 grid(1, 1, 1);

    switch (A.scalar_type()){
        case torch::kFloat16:
            wmma_m16n16k16_fp16_fp32_kernel<<<grid, block>>>(
                (half*)A.data_ptr<at::Half>(),
                (half*)B_col.data_ptr<at::Half>(),
                C.data_ptr<float>(),
                D.data_ptr<float>()
            );
            break;
        case torch::kBFloat16:
            wmma_m16n16k16_bf16_fp32_kernel<<<grid, block>>>(
                (::__nv_bfloat16*)A.data_ptr<at::BFloat16>(),
                (::__nv_bfloat16*)B_col.data_ptr<at::BFloat16>(),
                C.data_ptr<float>(),
                D.data_ptr<float>()
            );
            break;
        case torch::kFloat32:
            wmma_m16n16k8_tf32_fp32_kernel<<<grid, block>>>(
                A.data_ptr<float>(),
                B_col.data_ptr<float>(),
                C.data_ptr<float>(),
                D.data_ptr<float>()
            );
            break;
        default:
        TORCH_CHECK(false, "Unsupported dtype");
    }

    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_f16bf16tf32", &mma_f16bf16tf32, "GEMM (TF32/BF16/FP16->FP32)");
}
