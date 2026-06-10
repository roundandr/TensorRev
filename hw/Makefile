ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_ROOT := $(abspath $(ROOT)..)
include $(PROJECT_ROOT)/build_config.mk

NV_SETUP := $(ROOT)nv/gemm_setup.py
NV_BUILD_ENV := \
	TENSORREV_CUTLASS_DIR="$(CUTLASS_DIR)" \
	TENSORREV_CUDA_ARCH="$(CUDA_ARCH)"

.PHONY: all nv f8 nv/gemm_setup nv/gemm_setup_f8 \
	prepare_default_artifacts ensure_wmma_support ensure_cutlass_dir ensure_f8_support clean

all: nv

nv: nv/gemm_setup

f8: nv/gemm_setup_f8

nv/gemm_setup: prepare_default_artifacts ensure_wmma_support
	$(NV_BUILD_ENV) TENSORREV_BUILD_CUTLASS_GEMM_F8="0" $(PYTHON) $(NV_SETUP) build_ext --inplace

nv/gemm_setup_f8: ensure_cutlass_dir ensure_f8_support
	$(NV_BUILD_ENV) TENSORREV_BUILD_CUTLASS_GEMM_F8="1" $(PYTHON) $(NV_SETUP) build_ext --inplace

prepare_default_artifacts:
	rm -f $(ROOT)nv_mma*.so

ensure_wmma_support:
	@echo "Checking CUDA arch support for nv_wmma (requires sm70+; BF16/TF32 require sm80+)..."
	@$(NV_BUILD_ENV) $(PYTHON) $(NV_SETUP) --check-wmma-support

ensure_cutlass_dir:
	@if [ -z "$(strip $(CUTLASS_DIR))" ]; then \
		echo "CUTLASS_DIR is not configured."; \
		echo "Set CUTLASS_DIR in $(PROJECT_ROOT)/build_config.mk before building F8."; \
		exit 1; \
	fi
	@if [ ! -d "$(CUTLASS_DIR)/include" ]; then \
		echo "CUTLASS include directory not found: $(CUTLASS_DIR)/include"; \
		exit 1; \
	fi
	@if [ ! -d "$(CUTLASS_DIR)/tools/util/include" ]; then \
		echo "CUTLASS util include directory not found: $(CUTLASS_DIR)/tools/util/include"; \
		exit 1; \
	fi

ensure_f8_support:
	@echo "Checking CUDA arch support for cutlass_gemm_f8 (requires sm89+; sm89 requires CUDA 12.4+)..."
	@$(NV_BUILD_ENV) $(PYTHON) $(NV_SETUP) --check-f8-support

clean:
	rm -rf $(ROOT)build
	rm -f $(ROOT)*.so
