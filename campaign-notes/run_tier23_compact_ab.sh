#!/usr/bin/env bash
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile 2>/dev/null || true
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
BASE="${HOME}/Dendro-GR-runs/q8_compact_ab"
LOGDIR="${HOME}/Dendro-GR-notes/tier0"
SUMMARY="${LOGDIR}/tier23_compact_ab.log"
mkdir -p "${BASE}" "${LOGDIR}"
: > "${SUMMARY}"
echo "host $(hostname) $(date -Is)" | tee -a "${SUMMARY}"

run_one() {
  local tag="$1"
  local envextra="$2"
  local OUT="${BASE}/${tag}"
  local LOG="${LOGDIR}/tier23_${tag}.log"
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
  cd "${OUT}"
  echo "==== ${tag} ${envextra} ====" | tee -a "${SUMMARY}"
  START=$(date +%s)
  # shellcheck disable=SC2086
  mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 ${envextra} \
    "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
  END=$(date +%s)
  WALL=$((END-START))
  ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
  BD=$(grep -E 'COMPACT_BDY' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | head -4 || true)
  echo "tag=${tag} wall_sec=${WALL} ets=${ETS:-NA}" | tee -a "${SUMMARY}"
  echo "${BD}" | tee -a "${SUMMARY}"
}

run_one baseline "DENDRO_COMPACT_BDY=0"
run_one compact  "DENDRO_COMPACT_BDY=1"
echo TIER23_COMPACT_AB_DONE | tee -a "${SUMMARY}"
