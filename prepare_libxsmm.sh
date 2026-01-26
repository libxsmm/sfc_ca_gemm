#!/bin/bash
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

#
# Script to prepare/build LIBXSMM for SFC-CA GEMM standalone benchmark
#
# Usage:
#   ./prepare_libxsmm.sh [branch]
#
# Arguments:
#   branch: LIBXSMM git branch to use (default: main)
#
# Environment variables:
#   SFC_CA_GEMM_COMPILER: Compiler to use (gcc, clang, icx)
#   Default: gcc
#
###############################################################################

# Determine compiler to use
CC_USE=gcc
CXX_USE=g++

if [[ -z "$SFC_CA_GEMM_COMPILER" ]]; then
  CC_USE=gcc
  CXX_USE=g++
elif [[ "${SFC_CA_GEMM_COMPILER}" == "icx" ]]; then
  CC_USE=icx
  CXX_USE=icpx
elif [[ "${SFC_CA_GEMM_COMPILER}" == "clang" ]]; then
  CC_USE=clang
  CXX_USE=clang++
elif [[ "${SFC_CA_GEMM_COMPILER}" == "gcc" ]]; then
  CC_USE=gcc
  CXX_USE=g++
else
  CC_USE=gcc
  CXX_USE=g++
fi

# Default LIBXSMM branch
BRANCH=main
if [ $# -eq 1 ]; then
  BRANCH=$1
fi

echo "============================================================================"
echo "SFC-CA GEMM - LIBXSMM Preparation"
echo "============================================================================"
echo "Compiler:       ${CC_USE} / ${CXX_USE}"
echo "LIBXSMM branch: ${BRANCH}"
echo "============================================================================"
echo ""

# Clone or update LIBXSMM
if [ ! -d "libxsmm" ]; then
  echo "LIBXSMM not found, cloning from remote repository..."
  git clone https://github.com/libxsmm/libxsmm.git libxsmm
  cd libxsmm
else
  echo "LIBXSMM exists, updating..."
  cd libxsmm
  git pull
fi

echo ""
echo "Building LIBXSMM..."
echo "Switching to ${BRANCH} branch..."
git checkout $BRANCH
git pull

echo ""
echo "Compiling LIBXSMM (this may take a few minutes)..."
make realclean && make CC=${CC_USE} CXX=${CXX_USE} FC= -j16

if [ $? -eq 0 ]; then
  echo ""
  echo "============================================================================"
  echo "LIBXSMM successfully built!"
  echo "============================================================================"
  echo ""
  echo "To build SFC-CA GEMM benchmark:"
  echo "  make"
  echo ""
  echo "Or specify compiler explicitly:"
  echo "  SFC_CA_GEMM_COMPILER=gcc make"
  echo "  SFC_CA_GEMM_COMPILER=clang make"
  echo "  SFC_CA_GEMM_COMPILER=icx make"
  echo "============================================================================"
else
  echo ""
  echo "ERROR: LIBXSMM build failed!"
  echo "Please check the error messages above."
  exit 1
fi

cd ..
