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
- GPU architecture `sm_70` or newer for the default WMMA build

Optional for FP8 experiments:

- a local CUTLASS checkout
- GPU architecture `sm_89` or newer
- CUDA toolkit 12.4 or newer for `sm_89` FP8 builds

## Repository Layout

- [tensorrev/](./tensorrev): consolidated Python source directory
- [run.py](./run.py): top-level launcher script
- [tensorrev/mma_experiment.py](./tensorrev/mma_experiment.py): common MMA execution wrapper
- [tensorrev/accum_precision.py](./tensorrev/accum_precision.py): accumulation-precision detection
- [tensorrev/rounding.py](./tensorrev/rounding.py): rounding-mode detection
- [tensorrev/special_value.py](./tensorrev/special_value.py): special-value detection
- [tensorrev/common.py](./tensorrev/common.py): qualifier tables and dtype metadata
- [hw/](./hw): CUDA extension build logic and kernels

## Build

Default build:

```bash
make
```

This builds the non-F8 extension path.
Before compiling, the build checks that the target GPU architecture is at least `sm_70`.
For `sm_70`/`sm_75`, the WMMA extension builds only the FP16 path; for `sm_80+`, it also builds BF16 and TF32 paths.
If you run on Ada, Hopper, or Blackwell without building the F8 extension, the experiment driver will automatically fall back to Ampere qualifiers.

Optional FP8 build:

1. Edit [build_config.mk](./build_config.mk)
2. Set `CUTLASS_DIR` to your local CUTLASS checkout
3. Optionally set `CUDA_ARCH` if you do not want auto-detection
4. Run:

```bash
make f8
```

`make f8` checks that:

- the target GPU architecture is at least `sm_70` for the WMMA extension
- `CUTLASS_DIR` is configured correctly
- the target GPU architecture is at least `sm_89`

For F8 builds, TensorRev keeps `89` as `sm_89` in the generated `nvcc` `-gencode` flag. Plain `sm_90+` architecture targets such as `90`, `100`, and `110` are promoted to `90a`, `100a`, and `110a`. If you already specify an `a`-suffixed `sm_90+` target explicitly, it is preserved as-is.

## Run

Run the experiment suite from the repository root:

```bash
python run.py
```

The script will:

1. detect the current GPU name and compute capability
2. map that GPU to a TensorRev architecture
3. fall back to `Ampere` qualifiers automatically when the detected GPU would require F8 support but `make f8` has not been run
4. run all enabled experiments for each qualifier

## Example Results: NVIDIA Blackwell

The following results were collected on:

- GPU: `NVIDIA Thor`
- Compute capability: `11.0`

### Accumulation Precision and Rounding

| Qualifier | Accumulation Precision | Rounding Mode |
| --- | --- | --- |
| `m16n16k16.f32.f16.f16.f32` | `25` fraction bits | `RZ` |
| `m16n16k16.f32.tf32.tf32.f32` | `25` fraction bits | `RZ` |
| `m16n16k16.f32.bf16.bf16.f32` | `25` fraction bits | `RZ` |
| `m16n8k32.f32.e5m2.e5m2.f32` | `25` fraction bits | `RZ` |
| `m16n8k32.f32.e5m2.e4m3.f32` | `25` fraction bits | `RZ` |
| `m16n8k32.f32.e4m3.e5m2.f32` | `25` fraction bits | `RZ` |
| `m16n8k32.f32.e4m3.e4m3.f32` | `25` fraction bits | `RZ` |

### Subnormal Behavior

| Qualifier | `-0` | Subnormal C | Subnormal A | Subnormal B | Subnormal After Multiply | Subnormal After Add |
| --- | --- | --- | --- | --- | --- | --- |
| `m16n16k16.f32.f16.f16.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n16k16.f32.tf32.tf32.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n16k16.f32.bf16.bf16.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n8k32.f32.e5m2.e5m2.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n8k32.f32.e5m2.e4m3.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n8k32.f32.e4m3.e5m2.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n8k32.f32.e4m3.e4m3.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Overflow and NaN Behavior

| Qualifier | Overflow After Multiply | Overflow After Add | Overflow Edge Case | `+inf` from C | `-inf` from C | NaN A | NaN B | NaN `0 * inf` | NaN `-inf + inf` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `m16n16k16.f32.f16.f16.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n16k16.f32.tf32.tf32.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n16k16.f32.bf16.bf16.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n8k32.f32.e5m2.e5m2.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `m16n8k32.f32.e5m2.e4m3.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| `m16n8k32.f32.e4m3.e5m2.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| `m16n8k32.f32.e4m3.e4m3.f32` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |

### Notes on Interpretation

Accumulation precision here refers to the number of fraction bits only. It does not include the hidden bit, carry bit, or sign bit.

The accumulation-precision test uses:

```text
-1 + 1 + 2^t
```

starting from `t = 0` and increasing `t` until the output first deviates. The reported accumulation precision is then `t - 1`.

`e4m3` does not support `inf`, so the following cases are not applicable for `e4m3` inputs:

- `NaN from 0 * inf`
- `NaN from -inf + inf`
