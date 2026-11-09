#!/usr/bin/env bash
# A/B: unzip CTA 7x7 (baseline) vs 8x8 (DENDRO_UNZIP_TB88=1). q8/4GPU short.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT0="${HOME}/Dendro-GR-runs/q8_unzip_tb88"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier2_unzip_tb88_ab.log"
: > "${LOG}"
echo "host $(hostname) $(date -Is) unzip_tb88_ab" | tee -a "${LOG}"

run_one() {
  local tag="$1"; shift
  local envextra="$*"
  local OUT="${OUT0}/${tag}"
  rm -rf "${OUT}"; mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${OUT}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.5", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${OUT}/mesh_q8_tpid_sol.bin"
  cd "${OUT}"
  echo "==== ${tag} ${envextra} ====" | tee -a "${LOG}"
  # shellcheck disable=SC2086
  mpirun -np 4 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 ${envextra} \
    "${BIN}" "${OUT}/local.par.toml" 1 >> "${LOG}" 2>&1 || true
  grep -E 'ETS time|unzip-split' "${OUT}"/../"${tag}" 2>/dev/null || true
  grep 'ETS time' "${LOG}" | tail -1 | tee -a "${LOG}.summary" || true
}

run_one tb77 ""
run_one tb88 "DENDRO_UNZIP_TB88=1"
grep -E 'ETS time|====' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tee "${LOG}.summary"
echo TIER2_UNZIP_TB88_AB_DONE | tee -a "${LOG}.summary"
