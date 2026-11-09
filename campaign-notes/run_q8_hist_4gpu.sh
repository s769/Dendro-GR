#!/usr/bin/env bash
# q8 block-level hist on 4 GPUs (Tier 0.2 refresh / Tier 2.4 companion to q4).
set -euo pipefail
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
TPID_BIN="${DENDRO_BUILD}/BSSN_GR/tpid"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
OUT="${HOME}/Dendro-GR-runs/q8_hist_4gpu"
LOG="${HOME}/Dendro-GR-notes/tier0/q8_hist_4gpu.log"
rm -rf "${OUT}"
mkdir -p "${OUT}"/{aeh,cp,dat,vtu}

PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
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

cd "${OUT}"
if [[ ! -f "${PREFIX}_tpid_sol.bin" ]]; then
  echo "generating ${PREFIX}_tpid_sol.bin via tpid ..."
  "${TPID_BIN}" "${OUT}/local.par.toml" 4 > tpid.log 2>&1
fi
if [[ ! -f "${PREFIX}_tpid_sol.bin" ]]; then
  echo "NO_Q8_TPID after tpid"; ls -la; exit 1
fi

START=$(date +%s)
mpirun -np 4 env DENDRO_RHS_PT_ASYNC=1 DENDRO_DUMP_BLK_LEV=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
END=$(date +%s)
echo "wall_sec $((END-START))" | tee -a "${LOG}"
grep -E 'BLK_LEV|C_HAM|ETS time|Grid Info|wall_sec|level' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | head -80
echo Q8_HIST_4GPU_DONE
