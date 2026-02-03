#!/bin/bash

# Tuning script for K_layers and kbf parameters
# Tests various GEMM shapes and finds optimal configurations

OUTPUT_FILE="tune_results_$(date +%Y%m%d_%H%M%S).txt"

echo "Starting tuning sweep - output will be saved to ${OUTPUT_FILE}"
echo "============================================================" | tee -a ${OUTPUT_FILE}

# Define parameter ranges
MN_SIZES=(512 1024 2048 4096 8192)
K_SIZES=(576 1152 2304 4608 9216)
K_LAYERS_VALUES=(1 2 3 4 6 8)
KBF_VALUES=(1 2 3 4 6 8)

# Fixed parameters
BM=32
BN=32
BK=32
N_LAYERS=-1
N_ITERS=10
CHECK_CORRECTNESS=0

# Environment variables
export LIBXSMM_X86_AMX_GEMM_STREAMING_A=1
export LIBXSMM_X86_AMX_GEMM_STREAMING_B=1
export OMP_NUM_THREADS=96
export KMP_AFFINITY=granularity=fine,compact,1,0

# Total experiments counter
TOTAL_EXPERIMENTS=0
for M in "${MN_SIZES[@]}"; do
  for N in "${MN_SIZES[@]}"; do
    for K in "${K_SIZES[@]}"; do
      for K_LAYERS in "${K_LAYERS_VALUES[@]}"; do
        for KBF in "${KBF_VALUES[@]}"; do
          ((TOTAL_EXPERIMENTS++))
        done
      done
    done
  done
done

echo "Total experiments to run: ${TOTAL_EXPERIMENTS}" | tee -a ${OUTPUT_FILE}
echo "============================================================" | tee -a ${OUTPUT_FILE}

# Run experiments
EXPERIMENT_COUNT=0
for M in "${MN_SIZES[@]}"; do
  for N in "${MN_SIZES[@]}"; do
    for K in "${K_SIZES[@]}"; do
      echo "" | tee -a ${OUTPUT_FILE}
      echo "Testing M=${M}, N=${N}, K=${K}" | tee -a ${OUTPUT_FILE}
      echo "------------------------------------------------------------" | tee -a ${OUTPUT_FILE}
      
      for K_LAYERS in "${K_LAYERS_VALUES[@]}"; do
        for KBF in "${KBF_VALUES[@]}"; do
          ((EXPERIMENT_COUNT++))
          
          # Progress indicator
          echo -n "[$EXPERIMENT_COUNT/$TOTAL_EXPERIMENTS] Running: M=$M N=$N K=$K kbf=$KBF K_layers=$K_LAYERS ... "
          
          # Run the benchmark
          numactl -m 0,1,2 -C 0-95 ./sfc_ca_gemm $M $N $K $BM $BN $BK $KBF $K_LAYERS $N_LAYERS $N_ITERS $CHECK_CORRECTNESS >> ${OUTPUT_FILE} 2>&1
          
          if [ $? -eq 0 ]; then
            echo "Done"
          else
            echo "FAILED" | tee -a ${OUTPUT_FILE}
          fi
        done
      done
    done
  done
done

echo "" | tee -a ${OUTPUT_FILE}
echo "============================================================" | tee -a ${OUTPUT_FILE}
echo "All experiments completed!" | tee -a ${OUTPUT_FILE}
echo "============================================================" | tee -a ${OUTPUT_FILE}

# Process results to find best configurations
echo "" | tee -a ${OUTPUT_FILE}
echo "BEST CONFIGURATIONS:" | tee -a ${OUTPUT_FILE}
echo "============================================================" | tee -a ${OUTPUT_FILE}

SUMMARY_FILE="tune_summary_$(date +%Y%m%d_%H%M%S).txt"

# Extract all MEASURE lines and process them
grep "^MEASURE" ${OUTPUT_FILE} | while read -r line; do
  # Extract GFLOPS and configuration
  GFLOPS=$(echo $line | awk '{print $2}')
  CONFIG=$(echo $line | awk '{print $3}')
  
  # Extract M, N, K from configuration
  # Format: SFC_CA_GEMM_M_N_K_BM_BN_BK_bfKBF_replication_KLAYERS_threadsNUM_tileloadcomboX
  M_VAL=$(echo $CONFIG | cut -d'_' -f4)
  N_VAL=$(echo $CONFIG | cut -d'_' -f5)
  K_VAL=$(echo $CONFIG | cut -d'_' -f6)
  
  # Extract kbf and K_layers
  KBF_VAL=$(echo $CONFIG | grep -o 'bf[0-9]*' | sed 's/bf//')
  K_LAYERS_VAL=$(echo $CONFIG | grep -o 'replication_[0-9]*' | sed 's/replication_//')
  
  echo "${M_VAL}_${N_VAL}_${K_VAL} ${KBF_VAL} ${K_LAYERS_VAL} ${GFLOPS}" >> ${SUMMARY_FILE}.tmp
done

# Find best configuration for each M,N,K combination
for M in "${MN_SIZES[@]}"; do
  for N in "${MN_SIZES[@]}"; do
    for K in "${K_SIZES[@]}"; do
      KEY="${M}_${N}_${K}"
      
      # Find the line with maximum GFLOPS for this configuration
      BEST_LINE=$(grep "^${KEY} " ${SUMMARY_FILE}.tmp 2>/dev/null | sort -k4 -n -r | head -n1)
      
      if [ ! -z "$BEST_LINE" ]; then
        KBF_BEST=$(echo $BEST_LINE | awk '{print $2}')
        K_LAYERS_BEST=$(echo $BEST_LINE | awk '{print $3}')
        GFLOPS_BEST=$(echo $BEST_LINE | awk '{print $4}')
        
        printf "M=%5d N=%5d K=%5d | Best: kbf=%2d K_layers=%2d | GFLOPS=%.2f\n" \
          $M $N $K $KBF_BEST $K_LAYERS_BEST $GFLOPS_BEST | tee -a ${OUTPUT_FILE}
      fi
    done
  done
done

# Clean up temporary file
rm -f ${SUMMARY_FILE}.tmp

echo "" | tee -a ${OUTPUT_FILE}
echo "============================================================" | tee -a ${OUTPUT_FILE}
echo "Results saved to: ${OUTPUT_FILE}" | tee -a ${OUTPUT_FILE}
echo "============================================================" | tee -a ${OUTPUT_FILE}
