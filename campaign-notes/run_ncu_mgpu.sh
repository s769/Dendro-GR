#!/bin/bash
set -euo pipefail
module purge
module load cmake/4.4.0 gcc/15.3.0 cuda/13.2 ucc/1.9.0 openmpi/5.0.11
export GSL_ROOT_DIR=$HOME/software/gsl-2.8
export LD_LIBRARY_PATH=$GSL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
export PATH=/home1/apps/nvidia/Linux_aarch64/26.9/compilers/bin:$PATH
export OMP_NUM_THREADS=1
BIN=$HOME/Dendro-GR-build/BSSN_GR
OUT=$HOME/Dendro-GR/campaign-notes/ncu
mkdir -p "$OUT"
cd "$BIN"

DENDRO_TIME_PT_ASYNC=1 ./rhsgpu_test 128 1 0 >/dev/null 2>&1 || true

echo "=== NCU basic ==="
# Metric names without nested parens issues: use section instead
ncu --section LaunchStats --section Occupancy --launch-count 2 \
  -k regex:eval_rhs3 \
  -o "$OUT/ptasync_basic" --force-overwrite \
  env DENDRO_TIME_PT_ASYNC=1 ./rhsgpu_test 128 2 0 \
  > "$OUT/ptasync_ncu2.log" 2>&1 || true
tail -100 "$OUT/ptasync_ncu2.log" | tee "$OUT/ptasync_summary.txt"

echo "=== MULTI-GPU ==="
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g DENDRO_TIME_PT_ASYNC=1 ./rhsgpu_test 1024 10 0 > "$OUT/mgpu_g${g}.log" 2>&1 &
done
wait
for g in 0 1 2 3; do
  echo -n "GPU$g "
  grep -E 'pt_async_gpu_time' "$OUT/mgpu_g${g}.log" || tail -5 "$OUT/mgpu_g${g}.log"
done
echo DONE_NCU_MGPU
