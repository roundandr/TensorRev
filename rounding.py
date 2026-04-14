import torch
from mma_experiment import MmaExperiment


class RoundingExperiment(MmaExperiment):
    def __init__(self, arch: str, qualifier: str) -> None:
        super().__init__(arch=arch, qualifier=qualifier)
        self.a_min_norm_exp = self.a_min_exp + self.a_frac_bits
        self.b_min_norm_exp = self.b_min_exp + self.b_frac_bits
        self.sf = 1.0

    def build_case(self, *, case: str) -> bool:
        f = self.c_frac_bits

        self.reset()
        self.A[0, 0] = 1.0
        self.A[0, 1] = 1.0

        b_row1 = []
        b_row0 = [1.0, 1.0, -1.0, -1.0]

        if (-f - 2) >= self.b_min_norm_exp:
            scale = 2.0 ** -f
        else:
            scale_exp = self.b_min_norm_exp + 2
            scale = 2.0 ** scale_exp
            target_a00_exp = scale_exp + f
            self.sf = 2.0 ** target_a00_exp

            if target_a00_exp < self.a_max_exp:
                self.A[0, 0] = self.sf                
            else:
                overflow = target_a00_exp - self.a_max_exp
                if overflow > self.b_max_exp:
                    b0 = 2.0 ** self.b_max_exp
                    self.A[0, 1] = 2.0 ** -(overflow - self.b_max_exp)
                    self.sf = 2.0 ** (self.a_max_exp+self.b_max_exp)
                else:
                    b0 = 2.0 ** overflow
                b_row0 = [b0, b0, -b0, -b0]
                self.A[0, 0] = 2.0 ** self.a_max_exp

        if case == "rounding_mode":
            b_row1 = [0.75 * scale, 0.25 * scale, -0.75 * scale, -0.25 * scale]
        elif case == "tie_breaking_rule":
            b_row1 = [0.5 * scale, 0.5 * scale, -0.5 * scale, -0.5 * scale]
        else:
            raise ValueError(f"Unsupported case {case} for setup_rounding_case.")

        self.B[0, :4] = torch.tensor(b_row0, dtype=self.B.dtype, device="cuda")
        self.B[1, :4] = torch.tensor(b_row1, dtype=self.B.dtype, device="cuda")
        return True


    def set_ref(self, *, case: str):
        u = 2 ** (-self.c_frac_bits)
        expected = {
            "RU": [1 + u, 1 + u, -1.0, -1.0],
            "RD": [1.0, 1.0, -1 - u, -1 - u],
            "RZ": [1.0, 1.0, -1.0, -1.0],
            "RA": [1 + u, 1 + u, -1 - u, -1 - u],
            "RN": [1 + u, 1.0, -1 - u, -1.0],

            "RNU": [1 + u, 1 + 2.0 * u, -1.0, -1 - u],
            "RND": [1.0, 1 + u, -1.0 - u, -1 - 2.0 * u],
            "RNZ": [1.0, 1 + u, -1.0, -1 - u],
            "RNA": [1 + u, 1 + 2.0 * u, -1.0 - u, -1 - 2.0 * u],
            "RNE": [1.0, 1 + 2.0 * u, -1.0, -1 - 2.0 * u],
            "RNO": [1 + u, 1 + u, -1.0 - u, -1 - u],
        }
        self.ref[0, :4] = torch.tensor(expected[case], dtype=self.D.dtype, device="cuda") * self.sf
        return
    
    def detect(self) -> None:
        ROUNDING_MODE_LIST = ["RU", "RD", "RZ", "RA", "RN"]
        TIE_BREAKING_RULE_LIST = ["RNU", "RND", "RNZ", "RNA", "RNE", "RNO"]

        title = "Rounding Mode Detection Launch"
        border = "+" + "-" * (len(title) + 2) + "+"
        print(border)
        print(f"| {title} |")
        print(border)

        if(self.build_case(case="rounding_mode")):
            self.run()
            for rm in ROUNDING_MODE_LIST:
                self.set_ref(case=rm)
                if self.match(verbose=False):
                    self.print(name="D", indices=[(0, i) for i in range(4)])
                    print(f"Rounding mode detected: {rm}")
                    if rm == "RN":
                        self.build_case(case="tie_breaking_rule")
                        self.run()
                        for tbr in TIE_BREAKING_RULE_LIST:
                            self.set_ref(case=tbr)
                            if self.match(verbose=False):
                                print(f"Tie-breaking rule detected: {tbr}")
                                return
                        print("Tie-breaking rule is unknown.")
                    return
        print("Detect Falied! Rounding mode is unknown.")
        # self.print(name="D", indices=[(0, i) for i in range(4)])
        
