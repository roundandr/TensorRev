import torch
from .mma_experiment import MmaExperiment


class SpecialValueExperiment(MmaExperiment):
    def __init__(self, arch: str, qualifier: str, backend: str = "nv") -> None:
        super().__init__(arch=arch, qualifier=qualifier, backend=backend)
        self.a_max = (2 - 2**(-self.a_frac_bits)) * 2**self.a_max_exp
        self.b_max = (2 - 2**(-self.b_frac_bits)) * 2**self.b_max_exp


    def build_case(self, *, case: str) -> None:
        self.reset()
        handler_name = f"_build_{case}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            raise ValueError(f"Unsupported case {case}.")
        handler()

    def _build_subnormal_c_input(self) -> None:
        self.C[0, 0] = torch.finfo(self.C.dtype).tiny / 2.0
        self.ref[0, 0] = torch.finfo(self.C.dtype).tiny / 2.0

    def _build_subnormal_a_input(self) -> None:
        self.A[0, 0] = torch.finfo(self.A.dtype).tiny / 2.0
        self.B[0, 0] = 1.0
        self.ref[0, 0] = torch.finfo(self.A.dtype).tiny / 2.0

    def _build_subnormal_b_input(self) -> None:
        self.A[0, 0] = 1.0
        self.B[0, 0] = torch.finfo(self.B.dtype).tiny / 2.0
        self.ref[0, 0] = torch.finfo(self.B.dtype).tiny / 2.0

    def _build_subnormal_after_multiply(self) -> None:
        self.A[0, 0] = 0.5
        self.B[0, 0] = torch.finfo(self.B.dtype).tiny
        self.ref[0, 0] = torch.finfo(self.B.dtype).tiny / 2.0

    def _build_subnormal_after_add(self) -> None:
        self.A[0, 0] = 1.5
        self.B[0, 0] = torch.finfo(self.B.dtype).tiny
        self.A[0, 1] = -1.0
        self.B[1, 0] = torch.finfo(self.B.dtype).tiny
        self.ref[0, 0] = torch.finfo(self.B.dtype).tiny / 2.0

    def _build_negative_zero(self) -> None:
        self.C[0, 0] = -0.0
        self.A[0, 0] = +0.0
        self.B[0, 0] = -0.0
        self.ref[0, 0] = -0.0

    def _build_overflow_after_multiply(self) -> None:
        b_max = self.b_max
        self.A[0, 0] = 2.0
        self.B[0, 0] = b_max
        self.C[0, 0] = -b_max
        self.ref[0, 0] = b_max

    def _build_overflow_after_add(self) -> None:
        a_max = self.a_max
        self.A[0, 0] = a_max
        self.B[0, 0] = 1.0
        self.A[0, 1] = a_max
        self.B[1, 0] = 1.0
        self.C[0, 0] = -a_max
        self.ref[0, 0] = a_max

    def _build_overflow_edge_case(self) -> None:
        a_max = self.a_max
        E = self.b_max_exp
        u = 2 ** -self.b_frac_bits

        cases = [
            +(0.75 * u) * (2.0 ** E),
            +(0.50 * u) * (2.0 ** E),
            +(0.25 * u) * (2.0 ** E),
            -(0.75 * u) * (2.0 ** E),
            -(0.50 * u) * (2.0 ** E),
            -(0.25 * u) * (2.0 ** E),
        ]

        self.A[0, 0] = a_max
        self.A[0, 1] = 1
        for j, b in enumerate(cases):
            if(j < 3):
                self.B[0, j] = 1.0
                self.B[1, j] = b
                self.ref[0, j] = a_max + b
            else:
                self.B[0, j] = -1.0
                self.B[1, j] = b
                self.ref[0, j] = -a_max + b

    def _build_pos_inf_from_c(self) -> None:
        self.C[0, 0] = float("inf")
        self.ref[0, 0] = float("inf")

    def _build_neg_inf_from_c(self) -> None:
        self.C[0, 0] = float("-inf")
        self.ref[0, 0] = float("-inf")

    def _build_nan_a_input(self) -> None:
        self.A[0, 0] = float("nan")
        self.B[0, 0] = 0.0
        self.ref[0, :] = float("nan")

    def _build_nan_b_input(self) -> None:
        self.A[0, 0] = 0.0
        self.B[0, 0] = float("nan")
        self.ref[:, 0] = float("nan")

    def _build_nan_from_zero_times_inf(self) -> None:
        self.A[0, 0] = 0.0
        self.B[0, 0] = float("inf")
        self.ref[:, 0] = float("nan")

    def _build_nan_from_minus_inf_plus_inf(self) -> None:
        self.A[0, 0] = -1.0
        self.B[0, 0] = float("inf")
        self.A[0, 1] = +1.0
        self.B[1, 0] = float("inf")
        self.ref[:, 0] = float("nan")


    def detect(self) -> None:
        title = "Special Value Detection Launch"
        border = "+" + "-" * (len(title) + 2) + "+"
        print(border)
        print(f"| {title} |")
        print(border)

        CASE_ORDER = [
            "subnormal_c_input",
            "subnormal_a_input",
            "subnormal_b_input",
            "subnormal_after_multiply",
            "subnormal_after_add",
            "negative_zero",
            "overflow_after_multiply",
            "overflow_after_add",
            "overflow_edge_case",
            "pos_inf_from_c",
            "neg_inf_from_c",
            "nan_a_input",
            "nan_b_input",
        ]
        if self.a_type != "e4m3" and self.b_type != "e4m3":
            CASE_ORDER += [
                "nan_from_zero_times_inf",
                "nan_from_minus_inf_plus_inf",
            ]

        for case in CASE_ORDER:
            self.build_case(case=case)
            self.run()
            if not self.match(verbose=True):
                print(case, "Failed")
                print("=" * 50)
            else:
                print(case, "Passed")
            
