#!/usr/bin/env bash
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier21_f32su_lb1"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier21_f32su_lb1.log"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
rm -rf "${OUT}"; mkdir -p "${OUT}"
echo "host $(hostname) $(date -Is) f32su_lb1" | tee "${LOG}"
run_mode() {
  local mode="$1"
  local d="${OUT}/${mode}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${d}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.25", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 0", t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${d}/${PREFIX}_tpid_sol.bin"
  cd "${d}"
  case "${mode}" in
    ASYNC) env_rhs=(DENDRO_RHS_PT_ASYNC=1) ;;
    F32SU) env_rhs=(DENDRO_RHS_PT_F32SU=1) ;;
  esac
  echo "==== ${mode} ====" | tee -a "${LOG}"
  START=$(date +%s)
  mpirun -np 1 env -u CUDA_VISIBLE_DEVICES "${env_rhs[@]}" \
    "${BIN}" "${d}/local.par.toml" 1 >run.log 2>&1
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
  echo "${mode} wall=$((END-START))s ets=${ETS:-NA}" | tee -a "${LOG}"
}
run_mode ASYNC
run_mode F32SU
echo F32SU_LB1_DONE | tee -a "${LOG}"
