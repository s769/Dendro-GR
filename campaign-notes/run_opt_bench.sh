#!/bin/bash
set -euo pipefail
module purge
module load cmake/4.4.0 gcc/15.3.0 cuda/13.2 ucc/1.9.0 openmpi/5.0.11
export GSL_ROOT_DIR=$HOME/software/gsl-2.8
export LD_LIBRARY_PATH=$GSL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

BIN=$HOME/Dendro-GR-build/BSSN_GR
NOTES=$HOME/Dendro-GR/campaign-notes
hostname
nvidia-smi -L | head -1
cd "$BIN"

echo "=== MICROBENCH 2048 20 0 ==="
./rhsgpu_test 2048 20 0 2>&1 | tee "$NOTES/gb200_baseline2.txt" | grep -E 'gpu_time|l_inf var U_ALPHA|l_inf var U_CHI' | head -8
DENDRO_TIME_PT=1 ./rhsgpu_test 2048 20 0 2>&1 | tee "$NOTES/gb200_pt2.txt" | grep -E 'pt_gpu|l_inf var U_ALPHA|l_inf var U_CHI' | head -8
DENDRO_TIME_PT_ASYNC=1 ./rhsgpu_test 2048 20 0 2>&1 | tee "$NOTES/gb200_pt_async2.txt" | grep -E 'pt_async|l_inf var U_ALPHA|l_inf var U_CHI' | head -8
DENDRO_TIME_PT_DB=1 ./rhsgpu_test 2048 20 0 2>&1 | tee "$NOTES/gb200_pt_db2.txt" | grep -E 'pt_db|l_inf var U_ALPHA|l_inf var U_CHI' | head -8

echo "=== PT_DB PreferL1 (default) vs PreferShared ==="
DENDRO_TIME_PT_DB=1 ./rhsgpu_test 2048 20 0 2>&1 | tee "$NOTES/gb200_ptdb_preferL1.txt" | grep -E 'pt_db_gpu_time|l_inf var U_ALPHA' | head -4
DENDRO_TIME_PT_DB=1 DENDRO_PT_PREFER_SHARED=1 ./rhsgpu_test 2048 20 0 2>&1 | tee "$NOTES/gb200_ptdb_preferShared.txt" | grep -E 'pt_db_gpu_time|l_inf var U_ALPHA|invalid' | head -4

# PreferShared env may only wire for plain PT — check code path; also try without if PT_DB ignores it

echo "=== PT_ASYNC constraint smoke ==="
PAR=$HOME/Dendro-GR-runs/q1.smoke_cons.par.toml
RUN=$HOME/Dendro-GR-runs
TPID=$RUN/tpid_work/rit_q1_tpid_sol.bin
for pair in stock: ptasync:DENDRO_RHS_PT_ASYNC=1; do
  tag=${pair%%:*}
  envf=${pair#*:}
  dir=$RUN/smoke_cons_$tag
  rm -rf "$dir"
  mkdir -p "$dir"/{aeh,cp,dat,vtu}
  ln -sfn "$TPID" "$dir/rit_q1_tpid_sol.bin"
  cd "$dir"
  echo "--- $tag ---"
  T0=$(date +%s)
  if [[ -n "$envf" ]]; then
    env "$envf" "$BIN/bssnSolverCUDA" "$PAR" 1 > "$NOTES/evolve_cons_${tag}.log" 2>&1
  else
    "$BIN/bssnSolverCUDA" "$PAR" 1 > "$NOTES/evolve_cons_${tag}.log" 2>&1
  fi
  echo "WALL=$(( $(date +%s) - T0 ))"
  grep -E 'Constraint C_HAM|ETS time|U_ALPHA \(min' "$NOTES/evolve_cons_${tag}.log" \
    | sed 's/\x1b\[[0-9;]*m//g' | tail -12
done

python3 - <<'PY'
import pathlib
def load(tag):
    p = pathlib.Path(f"/home1/08435/srvenkat/Dendro-GR-runs/smoke_cons_{tag}/dat/bssn_profile_Constraints.dat")
    return [float(line.split()[2]) for i, line in enumerate(p.read_text().splitlines()) if i]
s, a = load("stock"), load("ptasync")
print("stock", s)
print("ptasync", a)
print("ptasync max|dHAM|", max(abs(x - y) for x, y in zip(s, a)))
PY
echo DONE_OPT
