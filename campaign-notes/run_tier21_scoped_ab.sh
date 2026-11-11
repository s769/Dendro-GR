#!/usr/bin/env bash
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
OUT="${HOME}/Dendro-GR-runs/tier21_scoped_ab"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier21_scoped_ab.log"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
TIME_END=0.25
mkdir -p "${OUT}"
echo "host $(hostname) $(date -Is) scoped_ab" | tee -a "${LOG}"
run_mode() {
  local mode="$1"
  local d="${OUT}/${mode}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${d}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = ${TIME_END}", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 0", t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${d}/${PREFIX}_tpid_sol.bin"
  cd "${d}"
  case "${mode}" in
    ASYNC) env_rhs=(DENDRO_RHS_PT_ASYNC=1) ;;
    SCOPED) env_rhs=(DENDRO_RHS_PT_ASYNC_DEFER=1) ;;
    *) return 1 ;;
  esac
  echo "==== ${mode} ====" | tee -a "${LOG}"
  START=$(date +%s)
  if ! mpirun -np 1 --bind-to none env -u CUDA_VISIBLE_DEVICES "${env_rhs[@]}" \
      "${BIN}" "${d}/local.par.toml" 1 >run.log 2>&1; then
    echo "FAIL ${mode}" | tee -a "${LOG}"
    tail -50 run.log | tee -a "${LOG}"
    return 1
  fi
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
  echo "${mode} wall=$((END-START))s ets=${ETS:-NA}" | tee -a "${LOG}"
}
# ASYNC already in log; only SCOPED if missing
if ! grep -q '^SCOPED wall=' "${LOG}" 2>/dev/null; then
  run_mode SCOPED
fi
echo SCOPED_AB_DONE | tee -a "${LOG}"
