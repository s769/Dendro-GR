#!/usr/bin/env bash
# Perpetual short ETS jobs until the SLURM alloc dies.
exec 9>"${HOME}/Dendro-GR-notes/tier0/keep_node_busy.lock"
if ! flock -n 9; then
  echo "$(date -Is) another keep_node_busy holds the lock; exit"
  exit 0
fi
set -uo pipefail
LOG="${HOME}/Dendro-GR-notes/tier0/keep_node_busy.log"
echo "$(date -Is) keep_node_busy start $(hostname)" | tee -a "$LOG"
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1 UCX_TLS="${UCX_TLS:-rc,sm,cuda_copy,cuda_ipc,gdr_copy}"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
n=0
while true; do
  n=$((n+1))
  case $((n % 3)) in
    1) TIME=0.5; PAR="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"; SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"; TAG=q8t05; SOLLINK=mesh_q8_tpid_sol.bin ;;
    2) TIME=0.5; PAR="${HOME}/Dendro-GR-runs/q4.meshprobe.par.toml"; SOL="${HOME}/Dendro-GR-runs/tpid_tier12/q4_a/mesh_q4_a_tpid_sol.bin"; TAG=q4t05; SOLLINK=$(basename "$SOL") ;;
    0) TIME=1.0; PAR="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"; SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"; TAG=q8t10; SOLLINK=mesh_q8_tpid_sol.bin ;;
  esac
  if [[ ! -f "$SOL" || ! -f "$PAR" ]]; then
    echo "$(date -Is) missing $PAR or $SOL" | tee -a "$LOG"
    sleep 30
    continue
  fi
  OUT="${HOME}/Dendro-GR-runs/keep_busy/${TAG}_c${n}"
  rm -rf "$OUT"
  mkdir -p "$OUT"/{aeh,cp,dat,vtu}
  cp "$PAR" "$OUT/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = ${TIME}", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
p.write_text(t)
PY
  ln -sfn "$SOL" "$OUT/$SOLLINK"
  cd "$OUT"
  echo "$(date -Is) cycle=$n tag=$TAG" | tee -a "$LOG"
  START=$(date +%s)
  mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
    "$BIN" "$OUT/local.par.toml" 1 \
    > "${HOME}/Dendro-GR-notes/tier0/keep_${TAG}_${n}.log" 2>&1 || true
  ETS=$(grep -E 'ETS time \(max\)' "${HOME}/Dendro-GR-notes/tier0/keep_${TAG}_${n}.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
  echo "tag=${TAG} n=${n} wall=$(( $(date +%s)-START )) ets=${ETS:-NA}" | tee -a "$LOG"
done
