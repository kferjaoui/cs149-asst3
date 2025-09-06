# -------- CUDA toolkit from the nvcc in PATH --------
CUDA_HOME ?= $(shell dirname $(shell dirname $(shell readlink -f $(shell which nvcc))))
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64

# -------- Variant selection --------
# Use: make baseline   or   make persistent
VAR ?= baseline

ifeq ($(VAR),baseline)
  CU_FILE     := saxpy.cu
  EXECUTABLE  := cudaSaxpy_baseline
  CPP_DEFS    := -DBUILD_BASELINE
else ifeq ($(VAR),persistent)
  CU_FILE     := saxpy_reuse_buffer.cu
  EXECUTABLE  := cudaSaxpy_persistent
  CPP_DEFS    := -DBUILD_PERSISTENT
else
  $(error Unknown VAR '$(VAR)'; use baseline or persistent)
endif

# -------- Toolchain/flags --------
ARCH    := $(shell uname -m)
CXX     := g++$(if $(filter-out aarch64,$(ARCH)), -m64)
NVCC    := nvcc

OBJDIR  := objs/$(VAR)
CXXFLAGS  := -O3 -Wall -I$(CUDA_INC)
NVCCFLAGS := -O3 -m64
LDFLAGS   := -L$(CUDA_LIB) -lcudart

# -------- Sources/objects --------
OBJ_MAIN := $(OBJDIR)/main.o
OBJ_CU   := $(OBJDIR)/$(basename $(notdir $(CU_FILE))).o
OBJS     := $(OBJ_MAIN) $(OBJ_CU)

# -------- Phony targets --------
.PHONY: default baseline persistent all clean run

default: $(EXECUTABLE)

baseline:
	@$(MAKE) VAR=baseline default

persistent:
	@$(MAKE) VAR=persistent default

all: baseline persistent

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -rf objs cudaSaxpy_* *.ppm *~

# -------- Build rules (explicit, no mixed implicit rules) --------
$(EXECUTABLE): $(OBJS) | $(OBJDIR)
	$(NVCC) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR):
	mkdir -p $@

$(OBJ_MAIN): main.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(CPP_DEFS) -c $< -o $@

$(OBJ_CU): $(CU_FILE) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
