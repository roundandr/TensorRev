import torch
from .accum_precision import AccumPrecisionExperiment
from .backend import resolve_backend
from .rounding import RoundingExperiment
from .special_value import SpecialValueExperiment
from .common import arch_mma_qualifiers, resolve_experiment_arch

class ExperimentImpl:
    def __init__(self, arch: str, qualifier: str, backend: str) -> None:
        self.arch = arch
        self.backend = backend
        self.qualifier = qualifier
        self.accum_precision_experiment = AccumPrecisionExperiment(
            arch=arch,
            qualifier=qualifier,
            backend=backend,
        )
        self.rounding_experiment = RoundingExperiment(
            arch=arch,
            qualifier=qualifier,
            backend=backend,
        )
        self.special_value_experiment = SpecialValueExperiment(
            arch=arch,
            qualifier=qualifier,
            backend=backend,
        )
    
    def run(self) -> None:
        self.accum_precision_experiment.detect()
        self.rounding_experiment.detect()
        self.special_value_experiment.detect()


def main() -> None:
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    backend = resolve_backend(device)
    detected_arch, fallback_arch = resolve_experiment_arch(device, backend)
    arch = fallback_arch or detected_arch

    print(f"GPU: {name}")
    print(f"Compute capability: {major}.{minor}")
    print(f"Selected TensorRev backend: {backend}")
    print(f"Detected TensorRev arch: {detected_arch}")
    if fallback_arch is not None:
        print("F8 extension `hw.nv_mma` is not available. Falling back to Ampere qualifiers.")
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

        experiment = ExperimentImpl(arch=arch, qualifier=qualifier, backend=backend)
        experiment.run()

        print("-" * 120)
        print(f"finished: {title}")
        print("-" * 120 + "\n")


if __name__ == "__main__":
    main()
