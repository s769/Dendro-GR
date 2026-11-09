#!/bin/bash
set -euo pipefail
module purge
module load cmake/4.4.0 gcc/15.3.0 cuda/13.2 ucc/1.9.0 openmpi/5.0.11
export GSL_ROOT_DIR=$HOME/software/gsl-2.8
export LD_LIBRARY_PATH=$GSL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
export PATH=/home1/apps/nvidia/Linux_aarch64/26.9/compilers/bin:$PATH
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BIN=$HOME/Dendro-GR-build/BSSN_GR
OUT=$HOME/Dendro-GR/campaign-notes/ncu
mkdir -p "$OUT"
cd "$BIN"
ncu --section LaunchStats --section Occupancy \
  --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
  --launch-count 2 -k regex:eval_rhs3_pt_async \
  -o "$OUT/ptasync_stalls" --force-overwrite \
  env DENDRO_TIME_PT_ASYNC=1 ./rhsgpu_test 2048 2 0 \
  > "$OUT/ptasync_stalls.log" 2>&1 || true
ncu --import "$OUT/ptasync_stalls.ncu-rep" 2>/dev/null \
  | tee "$OUT/ptasync_stalls_summary.txt" | head -220
echo DONE_NCU_PTASYNC
