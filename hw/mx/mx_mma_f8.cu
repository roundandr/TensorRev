#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v4f32 __attribute__((ext_vector_type(4)));

enum class F8Kind {
    E4M3,
    E5M2,
};

__device__ __forceinline__ int pack4_u8(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3) {
    uint32_t v =
        (static_cast<uint32_t>(x0) << 0)  |
        (static_cast<uint32_t>(x1) << 8)  |
        (static_cast<uint32_t>(x2) << 16) |
        (static_cast<uint32_t>(x3) << 24);
    return static_cast<int>(v);
}

template <F8Kind Kind>
__device__ __forceinline__ v4f32 builtin_mma(v2i32 a, v2i32 b, v4f32 c);

template <>
__device__ __forceinline__ v4f32 builtin_mma<F8Kind::E4M3>(
    v2i32 a, v2i32 b, v4f32 c) {
    return __builtin_mxc_mma_16x16x32f8(a, b, c);
}

template <>
__device__ __forceinline__ v4f32 builtin_mma<F8Kind::E5M2>(
    v2i32 a, v2i32 b, v4f32 c) {
    return __builtin_mxc_mma_16x16x32bf8(a, b, c);
}

template <F8Kind Kind>
__global__ void mma_m16n16k32_f8_builtin_kernel(
    const uint8_t* __restrict__ A,      // [16, 32], row-major contiguous
    const uint8_t* __restrict__ B_col,  // [16, 32], row-major contiguous = B.t().contiguous()
    const float* __restrict__ C,        // [16, 16], row-major contiguous
    float* __restrict__ D)              // [16, 16], row-major contiguous
{
    // warp size = 64
    int lane = threadIdx.x & 63;

    // A: logical shape [16, 32], row-major contiguous
    // lane 0..63 covers:
    //   row    = lane % 16      -> 16 rows
    //   kgroup = lane / 16      -> 4 groups, each group holds 8 elements
    int row = lane % 16;
    int kg  = lane / 16;   // 0..3
    int k0  = kg * 8;

    int a_base = row * 32 + k0;
    v2i32 fragA = {
        pack4_u8(A[a_base + 0], A[a_base + 1], A[a_base + 2], A[a_base + 3]),
        pack4_u8(A[a_base + 4], A[a_base + 5], A[a_base + 6], A[a_base + 7])
    };

    // B_col: expected shape [16, 32], row-major contiguous
    // It is produced by: B_col = B.t().contiguous(), where logical B is [32, 16]
    // Verified mapping: use the same lane decomposition as A, and load 8 contiguous bytes per lane
    int b_base = row * 32 + k0;
    v2i32 fragB = {
        pack4_u8(B_col[b_base + 0], B_col[b_base + 1], B_col[b_base + 2], B_col[b_base + 3]),
        pack4_u8(B_col[b_base + 4], B_col[b_base + 5], B_col[b_base + 6], B_col[b_base + 7])
    };

    // C/D: [16, 16] fp32
    int c_base = (lane / 16) * 64 + (lane % 16);
    v4f32 fragC = {
        C[c_base +  0],
        C[c_base + 16],
        C[c_base + 32],
        C[c_base + 48]
    };

    v4f32 acc = builtin_mma<Kind>(fragA, fragB, fragC);

    D[c_base +  0] = acc[0];
    D[c_base + 16] = acc[1];
    D[c_base + 32] = acc[2];
    D[c_base + 48] = acc[3];
}

torch::Tensor mma_f8(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B_col.is_cuda(), "B_col must be CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be CUDA tensor");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B_col.is_contiguous(), "B_col must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B_col.dim() == 2, "B_col must be 2D");
    TORCH_CHECK(C.dim() == 2, "C must be 2D");

    TORCH_CHECK(A.size(0) == 16 && A.size(1) == 32,
                "A must have shape [16, 32]");
    TORCH_CHECK(B_col.size(0) == 16 && B_col.size(1) == 32,
                "B_col must have shape [16, 32], e.g. B.t().contiguous() where B is [32, 16]");
    TORCH_CHECK(C.size(0) == 16 && C.size(1) == 16,
                "C must have shape [16, 16]");

    TORCH_CHECK(C.scalar_type() == at::ScalarType::Float,
                "C must be float32");

    auto a_type = A.scalar_type();
    auto b_type = B_col.scalar_type();

    TORCH_CHECK(
        a_type == at::ScalarType::Float8_e4m3fn || a_type == at::ScalarType::Float8_e5m2,
        "A must be float8_e4m3fn or float8_e5m2");
    TORCH_CHECK(
        b_type == at::ScalarType::Float8_e4m3fn || b_type == at::ScalarType::Float8_e5m2,
        "B_col must be float8_e4m3fn or float8_e5m2");

    TORCH_CHECK(
        a_type == b_type,
        "builtin version only supports A/B with the same float8 type "
        "(both e4m3 or both e5m2)");

    auto D = torch::empty_like(C);

    dim3 block(64, 1, 1);
    dim3 grid(1, 1, 1);

    auto* A_ptr = reinterpret_cast<const uint8_t*>(A.data_ptr());
    auto* B_ptr = reinterpret_cast<const uint8_t*>(B_col.data_ptr());
    auto* C_ptr = C.data_ptr<float>();
    auto* D_ptr = D.data_ptr<float>();

    if (a_type == at::ScalarType::Float8_e4m3fn) {
        mma_m16n16k32_f8_builtin_kernel<F8Kind::E4M3>
            <<<grid, block>>>(A_ptr, B_ptr, C_ptr, D_ptr);
    } else {
        mma_m16n16k32_f8_builtin_kernel<F8Kind::E5M2>
            <<<grid, block>>>(A_ptr, B_ptr, C_ptr, D_ptr);
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "mma_f8 kernel launch failed: ", cudaGetErrorString(err));

    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mma_f8", &mma_f8,
          "GEMM 16x16x32 f8/bf8 + fp32 accumulator (builtin, warp64)");
}
