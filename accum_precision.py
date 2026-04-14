import torch
from common import ACCUM_PRECISION_TEST_MODE
from mma_experiment import MmaExperiment

class AccumPrecisionExperiment(MmaExperiment):
    def __init__(self, arch: str, qualifier: str) -> None:
        super().__init__(arch=arch, qualifier=qualifier)
        self.ref_f64 = 0.0

    def build_case(self, t: int) -> None:
        self.reset()

        a_min_exp = self.a_min_exp
        b_min_exp = self.b_min_exp

        if(ACCUM_PRECISION_TEST_MODE == 1):
            a_min_exp = self.a_min_exp
            b_min_exp = self.b_min_exp
            a00 = 2.0 ** a_min_exp
            a01 = 2.0 ** a_min_exp
            b00 = 2.0 ** b_min_exp
            b10 = 2.0 ** (b_min_exp + t)

            self.A[0, 0] = a00
            self.A[0, 1] = a01
            self.B[0, 0] = b00
            self.B[1, 0] = b10

            self.ref_f64 = a00 * b00 + a01 * b10
        else:
            if -t >= a_min_exp:
                a00 = 2.0 ** (-t)
                a01 = -1.0
                a02 = 1.0

                b00 = 1.0
            else:
                a00 = 2.0 ** a_min_exp
                b00_exp = -t - a_min_exp
                if b00_exp >= b_min_exp:
                    b00 = 2.0 ** b00_exp
                    a01 = -1.0
                    a02 = 1.0
                else:
                    b00 = 2.0 ** b_min_exp
                    spill = t + b_min_exp + a_min_exp
                    a01 = -(2.0 ** spill)
                    a02 = 2.0 ** spill

            b10 = 1.0
            b20 = 1.0

            self.A[0, 0] = a00
            self.A[0, 1] = a01
            self.A[0, 2] = a02
            self.B[0, 0] = b00
            self.B[1, 0] = b10
            self.B[2, 0] = b20

            self.ref[0, 0] = a00 * b00 + a01 * b10 + a02 * b20
    
    def detect(self) -> None:
        title = "Accumulation Precision Detection Launch"
        border = "+" + "-" * (len(title) + 2) + "+"
        print(border)
        print(f"| {title} |")
        print(border)

        max_t = 100
        min_valid_t = 2
        for t in range(max_t):
            self.build_case(t=t)
            self.run()

            if ACCUM_PRECISION_TEST_MODE == 1:
                lhs = self.D[0, 0].to(torch.float64).item()
                rhs = self.ref_f64
                stop = lhs != rhs
                frac_bits = t + self.a_frac_bits - 1
            else:
                stop = not self.match(verbose=True)
                frac_bits = t - 1

            if not stop:
                continue
            
            if t < min_valid_t:
                print(f"Detection failed")
                print(f"Stop too early at t = {t}")
            else:
                print(f"Stop at iteration {t}")
                print(f"fraction bits for addition = {frac_bits}")
            return

        print("Detection failed")
        print(f"No stopping point found in t = [0, {max_t - 1}]")
        
