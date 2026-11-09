#!/usr/bin/env bash
# Tier 3.3 scout: cost vs initial separation at q=8 (proxy for hybridization length).
# Scales BH X coords (and TPID_PAR_B) by SEP_FAC while holding masses; short evolve.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier33_sep_cost_q8"
LOG="${HOME}/Dendro-GR-notes/tier0/tier33_sep_cost_q8.log"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
TIME_END=0.25

echo "host $(hostname) $(date -Is) tier33_sep_cost_q8" | tee "${LOG}"
[[ -x "${BIN}" && -f "${PAR0}" && -f "${SOL}" ]] || { echo "missing bin/par/sol"; exit 1; }
rm -rf "${OUT}"; mkdir -p "${OUT}"

run_fac() {
  local fac="$1"
  local tag="sep${fac}"
  local d="${OUT}/${tag}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
fac=float("${fac}")
p=Path("${d}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = ${TIME_END}", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 0", t)
# scale TPID_PAR_B and BH X (keep V as-is for this scout)
def scale_b(m):
    return m.group(1)+str(float(m.group(2))*fac)
t=re.sub(r"(TPID_PAR_B = )([0-9.eE+-]+)", scale_b, t)
def scale_bh_x(m):
    return m.group(1)+str(float(m.group(2))*fac)+m.group(3)
t=re.sub(r"(BSSN_BH[12] = \{ MASS = [0-9.eE+-]+, X = )([0-9.eE+-]+)(,)", scale_bh_x, t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${d}/${PREFIX}_tpid_sol.bin"
  cd "${d}"
  echo "==== ${tag} fac=${fac} ====" | tee -a "${LOG}"
  START=$(date +%s)
  mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
    "${BIN}" "${d}/local.par.toml" 1 >run.log 2>&1 || {
      echo "FAIL ${tag}" | tee -a "${LOG}"; tail -30 run.log | tee -a "${LOG}"; return 1; }
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
  LMAX=$(grep -E 'lmax:' run.log | tail -1 | sed 's/\x1b\[[0-9;]*m//g')
  echo "${tag} wall=$((END-START))s ets=${ETS:-NA} ${LMAX}" | tee -a "${LOG}"
}

run_fac 1.0
run_fac 1.25
echo TIER33_SEP_COST_Q8_DONE | tee -a "${LOG}"
