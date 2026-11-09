#!/usr/bin/env bash
# Tier 2.5: q8 strong scale 4GPU (1 node) vs 8GPU (2 nodes), CUDA-aware.
# Env: NODE1 NODE2 (default c125-022 c128-001)
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export UCX_TLS="${UCX_TLS:-rc,sm,cuda_copy,cuda_ipc,gdr_copy}"

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
BASE="${HOME}/Dendro-GR-runs/q8_scale_2node"
LOGDIR="${HOME}/Dendro-GR-notes/tier0"
SUMMARY="${LOGDIR}/tier25_2node_q8.log"
NODE1="${NODE1:-c125-022}"
NODE2="${NODE2:-c128-001}"
HOSTFILE="${BASE}/hostfile"
mkdir -p "${BASE}" "${LOGDIR}"
printf '%s slots=4\n%s slots=4\n' "${NODE1}" "${NODE2}" > "${HOSTFILE}"
: > "${SUMMARY}"
echo "host $(hostname) $(date -Is) 2node ${NODE1}+${NODE2}" | tee -a "${SUMMARY}"
cat "${HOSTFILE}" | tee -a "${SUMMARY}"

prep_out() {
  local OUT="$1"
  rm -rf "${OUT}"
  mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${OUT}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/local.par.toml")
t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.5", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_TIME_STEP_OUTPUT_FREQ = [0-9]+", "BSSN_TIME_STEP_OUTPUT_FREQ = 1", t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${OUT}/mesh_q8_tpid_sol.bin"
}

run_np4() {
  local OUT="${BASE}/np4"
  local LOG="${LOGDIR}/tier25_2node_np4.log"
  prep_out "${OUT}"
  cd "${OUT}"
  echo "==== np4 host=${NODE1}:4 ====" | tee -a "${SUMMARY}"
  START=$(date +%s)
  mpirun -np 4 --host "${NODE1}:4" \
    -x LD_LIBRARY_PATH -x PATH -x UCX_TLS \
    env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
    "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
  echo "tag=np4 wall_sec=$((END-START)) ets=${ETS:-NA}" | tee -a "${SUMMARY}"
}

run_np8() {
  local OUT="${BASE}/np8"
  local LOG="${LOGDIR}/tier25_2node_np8.log"
  prep_out "${OUT}"
  cd "${OUT}"
  echo "==== np8 hostfile ====" | tee -a "${SUMMARY}"
  START=$(date +%s)
  # Propagate module libs (libucc etc.) to remote ranks on the other idev.
  mpirun -np 8 --hostfile "${HOSTFILE}" --map-by ppr:4:node \
    -x LD_LIBRARY_PATH -x PATH -x UCX_TLS \
    env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
    "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
  echo "tag=np8 wall_sec=$((END-START)) ets=${ETS:-NA}" | tee -a "${SUMMARY}"
}

run_np4
run_np8
echo TIER25_2NODE_DONE | tee -a "${SUMMARY}"
