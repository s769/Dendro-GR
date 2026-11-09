#!/usr/bin/env bash
# Sweep BSSN_ASYNC_COMM_K (must divide BSSN_NUM_VARS=24) under deferred unzip sync.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export UCX_TLS="${UCX_TLS:-rc,sm,cuda_copy,cuda_ipc,gdr_copy}"

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
BASE="${HOME}/Dendro-GR-runs/q8_async_k_sweep"
LOGDIR="${HOME}/Dendro-GR/campaign-notes/tier0"
SUMMARY="${LOGDIR}/tier2_async_k_sweep.log"
mkdir -p "${BASE}" "${LOGDIR}"
: > "${SUMMARY}"
echo "host $(hostname) $(date -Is) async_k_sweep deferred_sync" | tee -a "${SUMMARY}"

run_one() {
  local k="$1"
  local tag="k${k}"
  local OUT="${BASE}/${tag}"
  local LOG="${LOGDIR}/tier2_async_k_${tag}.log"
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
t=re.sub(r"BSSN_ASYNC_COMM_K = [0-9]+", "BSSN_ASYNC_COMM_K = ${k}", t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${OUT}/mesh_q8_tpid_sol.bin"
  cd "${OUT}"
  echo "==== ${tag} ASYNC_COMM_K=${k} ====" | tee -a "${SUMMARY}"
  START=$(date +%s)
  mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 DENDRO_UNZIP_BATCH_SYNC=0 \
    "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
  echo "tag=${tag} wall_sec=$((END-START)) ets=${ETS:-NA}" | tee -a "${SUMMARY}"
  tail -5 "${LOG}" | grep -iE 'error|assert|divis|ETS' | tee -a "${SUMMARY}" || true
}

for k in 1 2 3 4 6 8; do
  run_one "$k"
done
echo TIER2_ASYNC_K_SWEEP_DONE | tee -a "${SUMMARY}"
