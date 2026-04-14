import torch
import importlib.util

volta_mma_qualifiers = [
    "m16n16k16.f32.f16.f16.f32",
]

ampere_mma_qualifiers = [
    "m16n16k16.f32.tf32.tf32.f32",
    "m16n16k16.f32.bf16.bf16.f32",
]

hopper_mma_qualifiers = [
    "m16n8k32.f32.e5m2.e5m2.f32",
    "m16n8k32.f32.e5m2.e4m3.f32",
    "m16n8k32.f32.e4m3.e5m2.f32",
    "m16n8k32.f32.e4m3.e4m3.f32",
]

blackwell_mma_qualifiers = [
    #TODO：Support Blackwell fp4 
    # "m16n8k32.f32.e5m2.e2m1.f32",
    # "m16n8k32.f32.e4m3.e2m1.f32",
    # "m16n8k32.f32.e2m1.e5m2.f32",
    # "m16n8k32.f32.e2m1.e4m3.f32",
    # "m16n8k32.f32.e2m1.e2m1.f32",
]

arch_mma_qualifiers = {
    "Volta": volta_mma_qualifiers,
    "Ampere": volta_mma_qualifiers + ampere_mma_qualifiers ,
    "Hopper": volta_mma_qualifiers + ampere_mma_qualifiers + hopper_mma_qualifiers,
    "Blackwell": volta_mma_qualifiers + 
                ampere_mma_qualifiers + 
                hopper_mma_qualifiers + 
                blackwell_mma_qualifiers,
}

FLOAT_DTYPE_SPECS = {
    "f64": {"frac_bits": 52, "exp_bits": 11, "min_exp": -1074, "max_exp": 1023},
    "f32": {"frac_bits": 23, "exp_bits": 8, "min_exp": -149, "max_exp": 127},
    "tf32": {"frac_bits": 10, "exp_bits": 8, "min_exp": -136, "max_exp": 127},
    "f16": {"frac_bits": 10, "exp_bits": 5, "min_exp": -24, "max_exp": 15},
    "bf16": {"frac_bits": 7, "exp_bits": 8, "min_exp": -133, "max_exp": 127},
    "e5m2": {"frac_bits": 2, "exp_bits": 5, "min_exp": -16, "max_exp": 15},
    "e4m3": {"frac_bits": 3, "exp_bits": 4, "min_exp": -9, "max_exp": 7},
}

BITS_VIEW_MAPPING = {
    torch.float16: torch.uint16,
    torch.bfloat16: torch.uint16,
    torch.float32: torch.uint32,
    torch.float64: torch.uint64,
    torch.int8: torch.uint8,
    torch.int16: torch.uint16,
    torch.int32: torch.uint32,
    torch.int64: torch.uint64,
}

nv_torch_dtype = {
    "f64": torch.float64,
    "f32": torch.float32,
    "tf32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
    # "ue8m0": torch.float8_e8m0fnu,
    "ue4m3": torch.float8_e4m3fn,
    "e2m1": torch.uint8,  # torch.float4_e2m1fn_x2 is not well-implemented
}

def nv_shape_to_mnk(shape: str) -> tuple[int, int, int]:
    mnk = shape.split("m")[1]
    m, nk = mnk.split("n")
    n, k = nk.split("k")
    return int(m), int(n), int(k)

def resolve_cuda_arch_name(device: int | None = None) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if device is None:
        device = torch.cuda.current_device()

    major, minor = torch.cuda.get_device_capability(device)
    capability = major * 10 + minor

    if capability >= 100:
        return "Blackwell"
    if capability >= 90:
        return "Hopper"
    if capability >= 80:
        return "Ampere"
    if capability >= 70:
        return "Volta"

    raise RuntimeError(f"Unsupported CUDA compute capability: {major}.{minor}")

def resolve_experiment_arch(device: int) -> tuple[str, str | None]:
    detected_arch = resolve_cuda_arch_name(device)
    if detected_arch not in {"Hopper", "Blackwell"}:
        return detected_arch, None
    if importlib.util.find_spec("hw.nv_mma") is not None:
        return detected_arch, None
    return detected_arch, "Ampere"

# Mode0: 1 - 1 + 2^t
# Mode1: min(subnormal)^2 + min(subnormal) * 2^t
ACCUM_PRECISION_TEST_MODE = 0 
