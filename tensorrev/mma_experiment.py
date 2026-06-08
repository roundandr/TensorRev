import torch
from typing import Iterable
from .backend import get_mma_function
from .common import (
    BITS_VIEW_MAPPING,
    FLOAT_DTYPE_SPECS,
    arch_mma_qualifiers,
    nv_torch_dtype,
    nv_shape_to_mnk,
)

class MmaExperiment:
    def __init__(self, arch: str, qualifier: str, backend: str = "nv") -> None:
        assert arch in arch_mma_qualifiers.keys(), (
            f"Unsupported architecture {arch} for mma.\n"
            f"Supported architectures: {list(arch_mma_qualifiers.keys())}"
        )
        supported_qualifiers = arch_mma_qualifiers[arch]
        assert qualifier in supported_qualifiers, (
            f"Unsupported qualifier {qualifier} for mma on {arch} architecture.\n"
            f"Supported qualifiers: {supported_qualifiers}"
        )
        shape, d_type, a_type, b_type, c_type = qualifier.split(".")

        self.arch = arch
        self.backend = backend
        self.qualifier = qualifier
        self.a_type = a_type
        self.b_type = b_type
        self.c_type = c_type
        self.d_type = d_type
        self.a_frac_bits = FLOAT_DTYPE_SPECS[self.a_type]["frac_bits"]
        self.b_frac_bits = FLOAT_DTYPE_SPECS[self.b_type]["frac_bits"]
        self.c_frac_bits = FLOAT_DTYPE_SPECS[self.c_type]["frac_bits"]
        self.a_max_exp = FLOAT_DTYPE_SPECS[self.a_type]["max_exp"]
        self.b_max_exp = FLOAT_DTYPE_SPECS[self.b_type]["max_exp"]
        self.a_min_exp = FLOAT_DTYPE_SPECS[self.a_type]["min_exp"]
        self.b_min_exp = FLOAT_DTYPE_SPECS[self.b_type]["min_exp"]

        m, n, k = nv_shape_to_mnk(shape)
        self.A = torch.zeros((m, k), dtype=nv_torch_dtype[a_type], device="cuda")
        self.B = torch.zeros((k, n), dtype=nv_torch_dtype[b_type], device="cuda")
        self.C = torch.zeros((m, n), dtype=nv_torch_dtype[c_type], device="cuda")
        self.D = torch.zeros((m, n), dtype=nv_torch_dtype[d_type], device="cuda")
        self.ref = torch.zeros((m, n), dtype=nv_torch_dtype[d_type], device="cuda")

    def reset(self) -> None:
        self.A.zero_()
        self.B.zero_()
        self.C.zero_()
        self.D.zero_()
        self.ref.zero_()

    def run(self) -> None:
        fn = get_mma_function(self.backend, self.A.dtype)
        self.D = fn(self.A, self.B.t().contiguous(), self.C)

    def match(
        self,
        *,
        verbose: bool = True,
    ) -> bool:
        lhs = self.D.detach()
        rhs = self.ref.detach().to(device=lhs.device)

        equal_mask = (lhs == rhs) | (torch.isnan(lhs) & torch.isnan(rhs))
        if torch.all(equal_mask):
            return True
        if verbose:
            mismatch_indices = torch.nonzero(~equal_mask, as_tuple=False)
            total = mismatch_indices.shape[0]
            mismatch_list = [tuple(mismatch_indices[i].tolist()) for i in range(total)]
            print(f"Found {total} mismatched element(s)")
            self.print(name="D", indices=mismatch_list)
            self.print(name="ref", indices=mismatch_list)
        return False

    def print(
            self, 
            name: str,
            indices: Iterable[int | tuple[int]] | None = None, 
        ) -> None: 
            tensor = {
                "A": self.A,
                "B": self.B,
                "C": self.C,
                "D": self.D,
                "ref": self.ref
            }[name]  
            if indices is None:
                print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
                print(tensor.cpu())
                return

            for index in indices:
                value = tensor[index]
                msg = f"{name}{index} = {value.item()} ({value.dtype})"
                bits_dtype = BITS_VIEW_MAPPING[value.dtype]
                if bits_dtype is not None:
                    bits = value.detach().contiguous().view(bits_dtype).item()
                    width = torch.iinfo(bits_dtype).bits // 4
                    msg += f", hex=0x{bits:0{width}X}"
                print(msg)
