PYTHON ?= python

# Set to your local CUTLASS checkout before building the F8 extension.
# Example:
# CUTLASS_DIR := /path/to/cutlass
CUTLASS_DIR ?=

# Use "auto" to detect from the current CUDA device capability.
# Override with values like 89, 90, 90a, 100, 110a, 120, or 120a when you need a specific target.
# For FP8 builds, sm89 uses 89 as-is and requires CUDA 12.4+; sm90+ targets are promoted to a-suffixed variants when needed.
# RTX 50-series / SM120 FP8 builds require CUDA 12.8+ and CUTLASS with SM120 builder support.
CUDA_ARCH ?= auto
