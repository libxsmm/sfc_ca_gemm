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
CC = gcc
CXX = g++
CFLAGS = -fopenmp -O3
CXXFLAGS = -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17 -O2

ifeq ($(SFC_CA_GEMM_COMPILER),gcc)
  CC := gcc
  CXX := g++
  CFLAGS := -fopenmp -O3
  CXXFLAGS := -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17 -O2 -Wno-vla-cxx-extension
endif

ifeq ($(SFC_CA_GEMM_COMPILER),clang)
  CC := clang
  CXX := clang++
  CFLAGS := -fopenmp=libomp -O3
  CXXFLAGS := -Wno-unused-command-line-argument -Wno-format -fopenmp=libomp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17 -O2 -Wno-vla-cxx-extension
endif

ifeq ($(SFC_CA_GEMM_COMPILER),icx)
  CC := icx
  CXX := icpx
  CFLAGS := -qopenmp -O3
  CXXFLAGS := -qopenmp -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17 -O2 -Wno-vla-cxx-extension
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
HEADERS = sfc_ca_gemm.hpp sfc_utils.h knn_model.h knn_model_emr.h knn_model_gnr.h roofline_predictor.h
KNN_MODEL_OBJ = knn_model.o
KNN_MODEL_EMR_OBJ = knn_model_emr.o
KNN_MODEL_GNR_OBJ = knn_model_gnr.o
ROOFLINE_PREDICTOR_OBJ = roofline_predictor.o

.PHONY: all
all: $(TARGET) $(TARGET_ONEDNN)

# Compile unified k-NN model interface
$(KNN_MODEL_OBJ): knn_model.c knn_model.h knn_model_emr.h knn_model_gnr.h
	$(CC) $(CFLAGS) -c knn_model.c -o $(KNN_MODEL_OBJ)

# Compile EMR-specific k-NN model
$(KNN_MODEL_EMR_OBJ): knn_model_emr.c knn_model_emr.h
	$(CC) $(CFLAGS) -c knn_model_emr.c -o $(KNN_MODEL_EMR_OBJ)

# Compile GNR-specific k-NN model
$(KNN_MODEL_GNR_OBJ): knn_model_gnr.c knn_model_gnr.h
	$(CC) $(CFLAGS) -c knn_model_gnr.c -o $(KNN_MODEL_GNR_OBJ)

# Compile roofline predictor
$(ROOFLINE_PREDICTOR_OBJ): roofline_predictor.c roofline_predictor.h
	$(CC) $(CFLAGS) -c roofline_predictor.c -o $(ROOFLINE_PREDICTOR_OBJ)

$(TARGET): $(SOURCES) $(HEADERS) $(KNN_MODEL_OBJ) $(KNN_MODEL_EMR_OBJ) $(KNN_MODEL_GNR_OBJ) $(ROOFLINE_PREDICTOR_OBJ)
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(ONEDNN_IFLAGS) $(SOURCES) $(KNN_MODEL_OBJ) $(KNN_MODEL_EMR_OBJ) $(KNN_MODEL_GNR_OBJ) $(ROOFLINE_PREDICTOR_OBJ) $(LFLAGS) $(LDFLAGS) -lxsmm -lm -o $@

$(TARGET_ONEDNN): $(SOURCES_ONEDNN) $(HEADERS) $(KNN_MODEL_OBJ) $(KNN_MODEL_EMR_OBJ) $(KNN_MODEL_GNR_OBJ) $(ROOFLINE_PREDICTOR_OBJ)
	$(CXX) $(CXXFLAGS) $(IFLAGS) $(ONEDNN_IFLAGS) $(SOURCES_ONEDNN) $(KNN_MODEL_OBJ) $(KNN_MODEL_EMR_OBJ) $(KNN_MODEL_GNR_OBJ) $(ROOFLINE_PREDICTOR_OBJ) $(LFLAGS) $(ONEDNN_LFLAGS) $(LDFLAGS) $(ONEDNN_LDFLAGS) -lxsmm -ldnnl -o $@

clean:
	rm -f $(TARGET) $(TARGET_ONEDNN) $(KNN_MODEL_OBJ) $(KNN_MODEL_EMR_OBJ) $(KNN_MODEL_GNR_OBJ) $(ROOFLINE_PREDICTOR_OBJ) *.o

.PHONY: all clean
