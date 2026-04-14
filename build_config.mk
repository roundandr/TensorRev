PYTHON ?= python

# Set to your local CUTLASS checkout before building the F8 extension.
# Example:
# CUTLASS_DIR := /path/to/cutlass
CUTLASS_DIR ?=

# Use "auto" to detect from the current CUDA device capability.
# Override with values like 90, 90a, 100, 110a when you need a specific target.
CUDA_ARCH ?= auto
