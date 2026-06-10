#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#ifndef TENSORREV_F8_ARCH_NUMBER
#define TENSORREV_F8_ARCH_NUMBER 90
#endif

#if TENSORREV_F8_ARCH_NUMBER == 89 && \
    (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4))
#error "sm89 FP8 builds require CUDA toolkit 12.4 or newer."
#endif

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

#if TENSORREV_F8_ARCH_NUMBER >= 120 && !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#error "sm120 FP8 builds require CUDA toolkit 12.8+ and CUTLASS with SM120 collective-builder support."
#endif

#if TENSORREV_F8_ARCH_NUMBER == 89
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#endif

#define CHECK_CUDA(x) TORCH_CHECK((x) == cudaSuccess, "CUDA error: ", cudaGetErrorString(x))

template <typename T>
struct TorchScalarType;

template <>
struct TorchScalarType<cutlass::float_e5m2_t> {
  static constexpr auto value = torch::kFloat8_e5m2;
  static constexpr const char* name = "torch.float8_e5m2";
};

template <>
struct TorchScalarType<cutlass::float_e4m3_t> {
  static constexpr auto value = torch::kFloat8_e4m3fn;
  static constexpr const char* name = "torch.float8_e4m3fn";
};

struct F8GemmProblem {
  int64_t M;
  int64_t N;
  int64_t K;
  torch::Tensor D;
};

template <typename ElementA, typename ElementB>
F8GemmProblem prepare_f8_gemm_problem(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B_col.is_cuda(), "B_col must be a CUDA tensor");
  TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");

  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B_col.dim() == 2, "B_col must be 2D");
  TORCH_CHECK(C.dim() == 2, "C must be 2D");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t Kb = B_col.size(1);
  int64_t N = B_col.size(0);

  TORCH_CHECK(K == Kb, "A.shape[1] must equal B_col.shape[1]");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C shape must be [M, N]");

  TORCH_CHECK(A.scalar_type() == TorchScalarType<ElementA>::value,
              "A must be ", TorchScalarType<ElementA>::name);
  TORCH_CHECK(B_col.scalar_type() == TorchScalarType<ElementB>::value,
              "B_col must be ", TorchScalarType<ElementB>::name);
  TORCH_CHECK(C.scalar_type() == torch::kFloat,
              "C must be torch.float32");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_col.is_contiguous(), "B_col must be contiguous");
  TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

  return {M, N, K, torch::zeros({M, N}, C.options())};
}

template <class ArchTag, typename ElementA, typename ElementB>
torch::Tensor mma_f8_impl_typed(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
  using namespace cute;

  auto problem = prepare_f8_gemm_problem<ElementA, ElementB>(A, B_col, C);
  int64_t M = problem.M;
  int64_t N = problem.N;
  int64_t K = problem.K;
  auto D = problem.D;

  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using OpClass = cutlass::arch::OpClassTensorOp;
  using MmaTileShape_MNK = cute::conditional_t<
      (ArchTag::kMinComputeCapability >= 120),
      Shape<_128, _128, _128>,
      Shape<_128, _128, _64>>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OpClass,
          MmaTileShape_MNK, ClusterShape_MNK,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute,
          ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto
      >::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OpClass,
          ElementA, LayoutA, AlignmentA,
          ElementB, LayoutB, AlignmentB,
          ElementAccumulator,
          MmaTileShape_MNK, ClusterShape_MNK,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto
      >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int L = 1;

  StrideA stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape((int)M, (int)K, L));
  StrideB stride_B = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape((int)N, (int)K, L));
  StrideC stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape((int)M, (int)N, L));
  StrideD stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape((int)M, (int)N, L));

  float alpha = 1.0f;
  float beta  = 1.0f;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {(int)M, (int)N, (int)K, L},
      {
        reinterpret_cast<ElementA*>(A.data_ptr()),
        stride_A,
        reinterpret_cast<ElementB*>(B_col.data_ptr()),
        stride_B
      },
      {
        {alpha, beta},
        reinterpret_cast<ElementC*>(C.data_ptr()),
        stride_C,
        reinterpret_cast<ElementD*>(D.data_ptr()),
        stride_D
      }
  };

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(args);

  auto workspace = torch::empty(
      {(long long)workspace_size},
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

  cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "can_implement failed: ", cutlassGetStatusString(status));

  status = gemm_op.initialize(args, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "initialize failed: ", cutlassGetStatusString(status));

  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "run failed: ", cutlassGetStatusString(status));

  CHECK_CUDA(cudaGetLastError());
  return D;
}

#if TENSORREV_F8_ARCH_NUMBER == 89
template <typename ElementA, typename ElementB>
torch::Tensor mma_f8_ada_impl_typed(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
  auto problem = prepare_f8_gemm_problem<ElementA, ElementB>(A, B_col, C);
  int64_t M = problem.M;
  int64_t N = problem.N;
  int64_t K = problem.K;
  auto D = problem.D;

  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int ElementsPerAccessD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using EpilogueOutputOp =
      cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
          cutlass::epilogue::thread::Identity,
          ElementD,
          ElementD,
          ElementsPerAccessD,
          ElementAccumulator,
          ElementCompute>;

  using Gemm = cutlass::gemm::device::GemmUniversalWithAbsMax<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm89,
      cutlass::gemm::GemmShape<128, 64, 128>,
      cutlass::gemm::GemmShape<64, 32, 128>,
      cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3,
      AlignmentA,
      AlignmentB>;

  typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{
      ElementCompute(1.0f),
      ElementCompute(1.0f)};
  typename Gemm::EpilogueOutputOp::Params epilogue_params{
      activation_params,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr};

  cutlass::gemm::GemmCoord problem_size((int)M, (int)N, (int)K);
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      /*batch_count=*/1,
      epilogue_params,
      reinterpret_cast<ElementA const*>(A.data_ptr()),
      reinterpret_cast<ElementB const*>(B_col.data_ptr()),
      reinterpret_cast<ElementC const*>(C.data_ptr()),
      reinterpret_cast<ElementD*>(D.data_ptr()),
      nullptr,
      nullptr,
      M * K,
      N * K,
      M * N,
      M * N,
      0,
      (int)K,
      (int)K,
      (int)N,
      (int)N,
      0};

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "can_implement failed: ", cutlassGetStatusString(status));

  size_t workspace_size = Gemm::get_workspace_size(args);
  auto workspace = torch::empty(
      {(long long)workspace_size},
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

  status = gemm_op.initialize(args, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "initialize failed: ", cutlassGetStatusString(status));

  status = gemm_op();
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "run failed: ", cutlassGetStatusString(status));

  CHECK_CUDA(cudaGetLastError());
  return D;
}

torch::Tensor mma_f8_ada_impl(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
  auto a_type = A.scalar_type();
  auto b_type = B_col.scalar_type();

  if (a_type == torch::kFloat8_e5m2 && b_type == torch::kFloat8_e5m2) {
    return mma_f8_ada_impl_typed<cutlass::float_e5m2_t, cutlass::float_e5m2_t>(A, B_col, C);
  }
  if (a_type == torch::kFloat8_e5m2 && b_type == torch::kFloat8_e4m3fn) {
    return mma_f8_ada_impl_typed<cutlass::float_e5m2_t, cutlass::float_e4m3_t>(A, B_col, C);
  }
  if (a_type == torch::kFloat8_e4m3fn && b_type == torch::kFloat8_e5m2) {
    return mma_f8_ada_impl_typed<cutlass::float_e4m3_t, cutlass::float_e5m2_t>(A, B_col, C);
  }
  if (a_type == torch::kFloat8_e4m3fn && b_type == torch::kFloat8_e4m3fn) {
    return mma_f8_ada_impl_typed<cutlass::float_e4m3_t, cutlass::float_e4m3_t>(A, B_col, C);
  }

  TORCH_CHECK(
      false,
      "Unsupported input dtypes. Supported combinations are:\n"
      "  (torch.float8_e5m2,   torch.float8_e5m2)\n"
      "  (torch.float8_e5m2,   torch.float8_e4m3fn)\n"
      "  (torch.float8_e4m3fn, torch.float8_e5m2)\n"
      "  (torch.float8_e4m3fn, torch.float8_e4m3fn)"
  );
}
#endif

template <class ArchTag>
torch::Tensor mma_f8_impl(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
  auto a_type = A.scalar_type();
  auto b_type = B_col.scalar_type();

  if (a_type == torch::kFloat8_e5m2 && b_type == torch::kFloat8_e5m2) {
    return mma_f8_impl_typed<ArchTag, cutlass::float_e5m2_t, cutlass::float_e5m2_t>(A, B_col, C);
  }
  if (a_type == torch::kFloat8_e5m2 && b_type == torch::kFloat8_e4m3fn) {
    return mma_f8_impl_typed<ArchTag, cutlass::float_e5m2_t, cutlass::float_e4m3_t>(A, B_col, C);
  }
  if (a_type == torch::kFloat8_e4m3fn && b_type == torch::kFloat8_e5m2) {
    return mma_f8_impl_typed<ArchTag, cutlass::float_e4m3_t, cutlass::float_e5m2_t>(A, B_col, C);
  }
  if (a_type == torch::kFloat8_e4m3fn && b_type == torch::kFloat8_e4m3fn) {
    return mma_f8_impl_typed<ArchTag, cutlass::float_e4m3_t, cutlass::float_e4m3_t>(A, B_col, C);
  }

  TORCH_CHECK(
      false,
      "Unsupported input dtypes. Supported combinations are:\n"
      "  (torch.float8_e5m2,   torch.float8_e5m2)\n"
      "  (torch.float8_e5m2,   torch.float8_e4m3fn)\n"
      "  (torch.float8_e4m3fn, torch.float8_e5m2)\n"
      "  (torch.float8_e4m3fn, torch.float8_e4m3fn)"
  );
}

torch::Tensor cutlass_gemm_f8(torch::Tensor A, torch::Tensor B_col, torch::Tensor C) {
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  int current_arch = prop.major * 10 + prop.minor;
  TORCH_CHECK(
      current_arch >= TENSORREV_F8_ARCH_NUMBER,
      "cutlass_gemm_f8 was built for sm_",
      TENSORREV_F8_ARCH_NUMBER,
      " but current device is sm_",
      current_arch
  );

#if TENSORREV_F8_ARCH_NUMBER >= 120
  return mma_f8_impl<cutlass::arch::Sm120>(A, B_col, C);
#elif TENSORREV_F8_ARCH_NUMBER >= 100
  return mma_f8_impl<cutlass::arch::Sm100>(A, B_col, C);
#elif TENSORREV_F8_ARCH_NUMBER >= 90
  return mma_f8_impl<cutlass::arch::Sm90>(A, B_col, C);
#elif TENSORREV_F8_ARCH_NUMBER == 89
  return mma_f8_ada_impl(A, B_col, C);
#else
  TORCH_CHECK(false, "Unsupported FP8 build arch: sm_", TENSORREV_F8_ARCH_NUMBER);
  return torch::Tensor();
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_gemm_f8", &cutlass_gemm_f8,
        "FP8 GEMM via CUTLASS (supports e5m2/e4m3)");
}
