import torch
from accum_precision import AccumPrecisionExperiment
from rounding import RoundingExperiment
from special_value import SpecialValueExperiment
from common import arch_mma_qualifiers, resolve_cuda_arch_name

class ExperimentImpl:
    def __init__(self, arch: str, qualifier: str) -> None:
        self.arch = arch
        self.qualifier = qualifier
        self.accum_precision_experiment = AccumPrecisionExperiment(
            arch=arch,
            qualifier=qualifier,
        )
        self.rounding_experiment = RoundingExperiment(
            arch=arch,
            qualifier=qualifier,
        )
        self.special_value_experiment = SpecialValueExperiment(
            arch=arch,
            qualifier=qualifier,
        )
    
    def run(self) -> None:
        self.accum_precision_experiment.detect()
        self.rounding_experiment.detect()
        self.special_value_experiment.detect()


if __name__ == "__main__":
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    arch = resolve_cuda_arch_name(device)

    print(f"GPU: {name}")
    print(f"Compute capability: {major}.{minor}")
    print(f"Selected TensorRev arch: {arch}")

    idx = 0
    qualifiers = arch_mma_qualifiers[arch]
    total = len(qualifiers)
    
    for qualifier in qualifiers:
        idx += 1
        title = f"[{idx}/{total}] arch={arch} | qualifier={qualifier}"

        print("\n" + "=" * 120)
        print(title)
        print("=" * 120)

        experiment = ExperimentImpl(arch=arch, qualifier=qualifier)
        experiment.run()

        print("-" * 120)
        print(f"finished: {title}")
        print("-" * 120 + "\n")

    
