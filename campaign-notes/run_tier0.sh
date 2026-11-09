#!/bin/bash
#SBATCH -J tier0_lts
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o /home1/08435/srvenkat/Dendro-GR/campaign-notes/tier0_%j.out
#SBATCH -e /home1/08435/srvenkat/Dendro-GR/campaign-notes/tier0_%j.err

set -euo pipefail
module purge
module load cmake/4.4.0 gcc/15.3.0 cuda/13.2 ucc/1.9.0 openmpi/5.0.11
export GSL_ROOT_DIR=$HOME/software/gsl-2.8
export LD_LIBRARY_PATH=/opt/apps/gcc/15.3.0/lib:$GSL_ROOT_DIR/lib:/home1/apps/nvidia/Linux_aarch64/26.9/compilers/lib:${LD_LIBRARY_PATH:-}
export PATH=/home1/apps/nvidia/Linux_aarch64/26.9/compilers/bin:$PATH
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

NOTES=$HOME/Dendro-GR/campaign-notes
BIN=$HOME/Dendro-GR-build/BSSN_GR
OUT=$NOTES/tier0
mkdir -p "$OUT" "$NOTES/nsys/small_nblocks"
hostname
nvidia-smi -L | head -1

echo "=== rebuild bssnSolverCUDA (BLK_LEV dump) ==="
cd $HOME/Dendro-GR-build
cmake --build . -j 16 --target bssnSolverCUDA rhsgpu_test 2>&1 | tail -30

echo "=== Tier 0.3 small-nblocks PT_ASYNC wall ==="
cd "$BIN"
for n in 32 64 128 256 512 2048; do
  t=$(DENDRO_TIME_PT_ASYNC=1 ./rhsgpu_test $n 20 0 2>/dev/null | grep pt_async_gpu_time | awk '{print $2}')
  echo "nblocks=$n time_s=$t" | tee -a "$OUT/small_nblocks_wall.txt"
done

echo "=== Tier 0.3 nsys + occupancy proxies ==="
for n in 32 64 128 256; do
  bash $NOTES/run_nsys_rhs.sh "tier0_nb${n}" ptasync $n 8 0 >"$OUT/nsys_${n}.log" 2>&1 || true
  rep=$(ls $NOTES/nsys/tier0_nb${n}_b0_*.nsys-rep 2>/dev/null | head -1 || true)
  if [[ -n "${rep:-}" ]]; then
    echo "---- cuda_api_sum n=$n ----" | tee -a "$OUT/nsys_api.txt"
    nsys stats --report cuda_api_sum "$rep" 2>/dev/null | head -25 | tee -a "$OUT/nsys_api.txt" || true
    echo "---- cuda_gpu_kern_sum n=$n ----" | tee -a "$OUT/nsys_kern.txt"
    nsys stats --report cuda_gpu_kern_sum "$rep" 2>/dev/null | head -20 | tee -a "$OUT/nsys_kern.txt" || true
  fi
done

# Occupancy via ncu on PT_ASYNC interior kernel at 32 and 256
if command -v ncu >/dev/null; then
  for n in 32 256; do
    ncu --set full --target-processes all -o "$OUT/ncu_nb${n}" \
      --kernel-name regex:bssnrhs3_pt_async \
      ./rhsgpu_test $n 2 0 >"$OUT/ncu_${n}.log" 2>&1 || \
    ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_blocks,sm__throughput.avg.pct_of_peak_sustained_elapsed \
      --target-processes all -o "$OUT/ncu_nb${n}_m" \
      ./rhsgpu_test $n 2 0 >"$OUT/ncu_${n}_m.log" 2>&1 || true
  done
fi

echo "=== Tier 0.2 mesh probes q4 / q8 ==="
for q in 4 8; do
  dir=$HOME/Dendro-GR-runs/meshprobe_q${q}
  rm -rf "$dir"
  mkdir -p "$dir"/{aeh,cp,dat,vtu}
  cd "$dir"
  echo "--- q=$q ---"
  DENDRO_DUMP_BLK_LEV=1 DENDRO_RHS_PT_ASYNC=1 \
    "$BIN/bssnSolverCUDA" "$HOME/Dendro-GR-runs/q${q}.meshprobe.par.toml" 1 \
    >"$OUT/mesh_q${q}.log" 2>&1 || true
  grep -E 'BLK_LEV_HIST|Grid Info \(After|lmin:|Remesh triggered' "$OUT/mesh_q${q}.log" \
    | sed 's/\x1b\[[0-9;]*m//g' | tee -a "$OUT/mesh_hist_summary.txt" || true
done

echo DONE_TIER0
