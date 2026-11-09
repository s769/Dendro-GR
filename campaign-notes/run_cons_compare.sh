#!/bin/bash
set -euo pipefail
module purge
module load cmake/4.4.0 gcc/15.3.0 cuda/13.2 ucc/1.9.0 openmpi/5.0.11
export GSL_ROOT_DIR=$HOME/software/gsl-2.8
export LD_LIBRARY_PATH=$GSL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

BIN=$HOME/Dendro-GR-build/BSSN_GR
PAR=$HOME/Dendro-GR-runs/q1.smoke_cons.par.toml
RUN=$HOME/Dendro-GR-runs
TPID=$RUN/tpid_work/rit_q1_tpid_sol.bin
NOTES=$HOME/Dendro-GR/campaign-notes

hostname
nvidia-smi -L | head -1
ls -la "$BIN/bssnSolverCUDA" "$PAR" "$TPID"

for pair in stock: ptdb:DENDRO_RHS_PT_DB=1 pt:DENDRO_RHS_PT=1; do
  tag=${pair%%:*}
  envf=${pair#*:}
  dir=$RUN/smoke_cons_$tag
  rm -rf "$dir"
  mkdir -p "$dir"/{aeh,cp,dat,vtu}
  ln -sfn "$TPID" "$dir/rit_q1_tpid_sol.bin"
  cd "$dir"
  echo "=== $tag ==="
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
    rows = []
    for i, line in enumerate(p.read_text().splitlines()):
        if i == 0:
            continue
        parts = line.split()
        rows.append(float(parts[2]))
    return rows
for t in ["stock", "pt", "ptdb"]:
    print(t, load(t))
s = load("stock")
for t in ["pt", "ptdb"]:
    d = load(t)
    print(t, "max|dHAM|", max(abs(a - b) for a, b in zip(s, d)))
PY
echo DONE_CONS
