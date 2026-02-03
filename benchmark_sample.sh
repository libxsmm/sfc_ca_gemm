#!/bin/bash

# Environment settings
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=64
#export LIBXSMM_X86_AMX_GEMM_STREAMING_A=1
#export LIBXSMM_X86_AMX_GEMM_STREAMING_B=1

# Default parameters
BM=32  # Block size M
BN=32  # Block size N
BK=32  # Block size K
N_LAYERS=-1  # n_layers parameter
N_ITERS=10  # Number of iterations
CHECK=0  # Check correctness (0=no, 1=yes)

# M=512 cases
echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 512 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 512 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 2048 $BM $BN $BK 4 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 2048 $BM $BN $BK 4 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 4096 $BM $BN $BK 4 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 4096 $BM $BN $BK 4 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 8192 $BM $BN $BK 4 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 512 8192 $BM $BN $BK 4 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 1024 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 1024 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 8192 $BM $BN $BK 2 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 1024 8192 $BM $BN $BK 2 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 2048 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 2048 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 2048 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 4096 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 512 8192 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

# M=1024 cases
echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 512 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 512 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 1024 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 1024 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 8192 $BM $BN $BK 2 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 512 8192 $BM $BN $BK 2 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 4096 $BM $BN $BK 2 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 4096 $BM $BN $BK 2 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 8192 $BM $BN $BK 4 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 1024 8192 $BM $BN $BK 4 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 2048 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 2048 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 2048 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 4096 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 8192 $BM $BN $BK 4 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 1024 8192 8192 $BM $BN $BK 4 2 $N_LAYERS $N_ITERS $CHECK

# M=2048 cases
echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 512 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 2048 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 1024 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 2048 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 2048 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 4096 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 2048 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 4096 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 4096 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 4096 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 8192 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 2048 8192 8192 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

# M=4096 cases
echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 1024 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 1024 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 2048 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 2048 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 512 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 2048 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 2048 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 1024 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 512 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 4096 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 4096 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 2048 8192 $BM $BN $BK 1 8 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 4096 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 4096 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 8192 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 4096 8192 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 8192 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 4096 8192 8192 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK

# M=8192 cases
echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 1024 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 4096 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 4096 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 8192 $BM $BN $BK 4 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 512 8192 $BM $BN $BK 4 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 1024 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 2048 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 8192 $BM $BN $BK 4 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 1024 8192 $BM $BN $BK 4 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 4096 $BM $BN $BK 1 4 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 8192 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 2048 8192 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 2048 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 4096 $BM $BN $BK 1 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 8192 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 4096 8192 $BM $BN $BK 2 2 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 512 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 1024 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 2048 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 2048 $BM $BN $BK 1 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 4096 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 4096 $BM $BN $BK 2 1 $N_LAYERS $N_ITERS $CHECK

echo "numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 8192 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK"
numactl -m 0 -C 0-63 ./sfc_ca_gemm 8192 8192 8192 $BM $BN $BK 4 1 $N_LAYERS $N_ITERS $CHECK
