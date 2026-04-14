# TensorRev

TensorRev is a small CUDA/PyTorch experiment suite for probing MMA behavior on NVIDIA GPUs. It currently focuses on three categories of behavior:

- accumulation precision
- rounding behavior
- special-value handling

The experiment driver selects the TensorRev architecture automatically from the current GPU compute capability and runs the supported MMA qualifiers for that architecture.

## Requirements

- Python with `torch` installed
- CUDA-capable NVIDIA GPU
- A working CUDA toolchain for building the extensions
- GNU Make
- GPU architecture `sm_80` or newer for the default WMMA build

Optional for FP8 experiments:

- a local CUTLASS checkout
- GPU architecture `sm_90` or newer

## Repository Layout

- [experiment_impl.py](./experiment_impl.py): top-level experiment entrypoint
- [mma_experiment.py](./mma_experiment.py): common MMA execution wrapper
- [accum_precision.py](./accum_precision.py): accumulation-precision detection
- [rounding.py](./rounding.py): rounding-mode detection
- [special_value.py](./special_value.py): special-value detection
- [common.py](./common.py): qualifier tables and dtype metadata
- [hw/](./hw): CUDA extension build logic and kernels

## Build

Default build:

```bash
make
```

This builds the non-F8 extension path.
Before compiling, the build checks that the target GPU architecture is at least `sm_80`, because `wmma_f16bf16tf32` requires `sm_80+`.

Optional FP8 build:

1. Edit [build_config.mk](./build_config.mk)
2. Set `CUTLASS_DIR` to your local CUTLASS checkout
3. Optionally set `CUDA_ARCH` if you do not want auto-detection
4. Run:

```bash
make f8
```

`make f8` checks that:

- the target GPU architecture is at least `sm_80` for the WMMA extension
- `CUTLASS_DIR` is configured correctly
- the target GPU architecture is at least `sm_90`

## Run

Run the experiment suite from the repository root:

```bash
python experiment_impl.py
```

The script will:

1. detect the current GPU name and compute capability
2. map that GPU to a TensorRev architecture
3. select the corresponding `arch_mma_qualifiers`
4. run all enabled experiments for each qualifier

