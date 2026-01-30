###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Evangelos Georganas (Intel Corp.)
###############################################################################

# Standalone SFC-based Communication-Avoiding GEMM Benchmark
# Minimal dependencies: only libxsmm required

BLDDIR = .
OUTDIR = .

# LIBXSMM paths - adjust these to your installation or use prepare_libxsmm.sh
LIBXSMM_ROOT := $(if $(LIBXSMM_ROOT),$(LIBXSMM_ROOT),./libxsmm)

# Compiler selection via SFC_CA_GEMM_COMPILER environment variable
# Options: gcc (default), clang, icx
CXX = g++
CXXFLAGS = -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2

ifeq ($(SFC_CA_GEMM_COMPILER),gcc)
  CXX := g++
  CXXFLAGS := -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 -Wno-vla-cxx-extension
endif

ifeq ($(SFC_CA_GEMM_COMPILER),clang)
  CXX := clang++
  CXXFLAGS := -Wno-unused-command-line-argument -Wno-format -fopenmp=libomp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 -Wno-vla-cxx-extension
endif

ifeq ($(SFC_CA_GEMM_COMPILER),icx)
  CXX := icpx
  CXXFLAGS := -qopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -O2 -Wno-vla-cxx-extension
endif

LDFLAGS = -Wl,-rpath,'$(LIBXSMM_ROOT)/lib'
IFLAGS = -I$(LIBXSMM_ROOT)/include
LFLAGS = -L$(LIBXSMM_ROOT)/lib

# oneDNN paths - set ONEDNN_ROOT environment variable or adjust these paths
# Example: export ONEDNN_ROOT=/path/to/onednn
ONEDNN_ROOT ?= $(HOME)/onednn_2026/oneDNN
ONEDNN_IFLAGS = -I$(ONEDNN_ROOT)/include -I$(ONEDNN_ROOT)/build/include
ONEDNN_LFLAGS = -L$(ONEDNN_ROOT)/build/src
ONEDNN_LDFLAGS = -Wl,-rpath,$(ONEDNN_ROOT)/build/src

TARGET = sfc_ca_gemm
TARGET_ONEDNN = sfc_ca_onednn_gemm
SOURCES = sfc_ca_gemm.cpp
SOURCES_ONEDNN = sfc_ca_onednn_gemm.cpp
HEADERS = sfc_ca_gemm.hpp sfc_utils.h

.PHONY: all
all: $(TARGET) $(TARGET_ONEDNN)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(SOURCES) $(LFLAGS) $(LDFLAGS) -lxsmm -o $@

$(TARGET_ONEDNN): $(SOURCES_ONEDNN) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(ONEDNN_IFLAGS) $(SOURCES_ONEDNN) $(LFLAGS) $(ONEDNN_LFLAGS) $(LDFLAGS) $(ONEDNN_LDFLAGS) -lxsmm -ldnnl -o $@

clean:
	rm -f $(TARGET) $(TARGET_ONEDNN) *.o

.PHONY: all clean
