#!/usr/bin/env bash
# Short ETS cost table for re-TPID'd TARGET_ORBITS sols (q1).
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier33_cost_orbits.log"
TAGS="${*:-N3 N5 N8}"

mkdir -p "$(dirname "${LOG}")"
echo "host $(hostname) $(date -Is) cost orbits: ${TAGS}" | tee "${LOG}"
test -x "${BIN}"

for tag in ${TAGS}; do
  sol="${HOME}/Dendro-GR-runs/tpid_tier33_orbits/${tag}/rit_q1_${tag}_tpid_sol.bin"
  par_src="${HOME}/Dendro-GR-runs/tpid_tier33_orbits/${tag}/local.par.toml"
  if [[ ! -f "${sol}" ]]; then
    echo "missing ${sol}" | tee -a "${LOG}"
    continue
  fi
  Nf=$(grep '^BSSN_TARGET_ORBITS' "${par_src}" | awk '{print $3}')
  d="${HOME}/Dendro-GR-runs/tier33_cost_${tag}"
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${PAR0}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p = Path("${d}/local.par.toml")
t = p.read_text()
t = re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.15", t)
t = re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t = re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
n = float("${Nf}")
line = f"BSSN_TARGET_ORBITS = {n}"
if "BSSN_TARGET_ORBITS" in t:
    t = re.sub(r"BSSN_TARGET_ORBITS = [0-9.eE+-]+", line, t)
else:
    t = line + "\n" + t
t = re.sub(r'TPID_FILEPREFIX = "[^"]*"', f'TPID_FILEPREFIX = "rit_q1_${tag}"', t)
p.write_text(t)
print("wrote", p, line)
PY
  ln -sfn "${sol}" "${d}/rit_q1_${tag}_tpid_sol.bin"
  cd "${d}"
  echo "==== cost ${tag} N=${Nf} ====" | tee -a "${LOG}"
  START=$(date +%s)
  if ! mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
      "${BIN}" local.par.toml 1 >run.log 2>&1; then
    echo "FAIL_${tag}" | tee -a "${LOG}"
    tail -30 run.log | tee -a "${LOG}"
    continue
  fi
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
  LMAX=$(grep -E 'lmax:' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 || true)
  echo "${tag} wall=$((END-START))s ets=${ETS:-NA} ${LMAX}" | tee -a "${LOG}"
done
echo T33_COST_ORBITS_DONE | tee -a "${LOG}"
