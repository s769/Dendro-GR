#!/bin/bash
# HANDOFF_NEXT item B: stall / bandwidth profile of PT_F32SU
set -euo pipefail
module purge
module load cmake/4.4.0 gcc/15.3.0 cuda/13.2 ucc/1.9.0 openmpi/5.0.11
export GSL_ROOT_DIR=$HOME/software/gsl-2.8
export LD_LIBRARY_PATH=$GSL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
export PATH=/home1/apps/nvidia/Linux_aarch64/26.9/compilers/bin:$PATH
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BIN=$HOME/Dendro-GR-build/BSSN_GR
OUT=$HOME/Dendro-GR-notes/ncu
mkdir -p "$OUT"
cd "$BIN"

echo "=== ptxas-style resource dump via cuobjdump / nvcc verbose rebuild note ==="
# Kernel resources from NCU LaunchStats; also try cuobjdump if available
cuobjdump -res-usage ./rhsgpu_test 2>/dev/null | grep -A2 -i 'eval_rhs3_pt_f32su' | head -40 \
  | tee "$OUT/ptf32su_res_usage.txt" || true

echo "=== NCU stalls + throughput ==="
# Sections covering stalls, memory, compute
ncu --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --launch-count 2 \
  -k regex:eval_rhs3_pt_f32su \
  -o "$OUT/ptf32su_stalls" --force-overwrite \
  env DENDRO_TIME_PT_F32SU=1 ./rhsgpu_test 2048 2 0 \
  > "$OUT/ptf32su_stalls.log" 2>&1 || true

ncu --import "$OUT/ptf32su_stalls.ncu-rep" 2>/dev/null \
  | tee "$OUT/ptf32su_stalls_summary.txt" | head -250

echo DONE_B_NCU
