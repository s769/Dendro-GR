#!/usr/bin/env bash
# Short q4 evolve on 4 GPUs for wall-time / constraint sample (Tier 2.4 lite).
set -euo pipefail
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q4.meshprobe.par.toml"
TPID_DIR="${HOME}/Dendro-GR-runs/tpid_tier12/q4_a"
OUT="${HOME}/Dendro-GR-runs/q4_short_4gpu"
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/q4_short_4gpu.log"
rm -rf "${OUT}"
mkdir -p "${OUT}"/{aeh,cp,dat,vtu}

# Need a q4 tpid sol — use meshprobe prefix or generate quickly if missing
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
if [[ ! -f "${OUT}/${PREFIX}_tpid_sol.bin" ]]; then
  if [[ -f "${TPID_DIR}/mesh_q4_a_tpid_sol.bin" ]]; then
    ln -sfn "${TPID_DIR}/mesh_q4_a_tpid_sol.bin" "${OUT}/${PREFIX}_tpid_sol.bin"
  else
    cd "${OUT}"
    cp "${PAR0}" local.par.toml
    # shorten for tpid path only
    "${DENDRO_BUILD}/BSSN_GR/tpid" local.par.toml 4 > tpid.log 2>&1
  fi
fi

cp "${PAR0}" "${OUT}/local.par.toml"
# short evolve: few remesh tests
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
ln -sfn "$(readlink -f "${OUT}/${PREFIX}_tpid_sol.bin" 2>/dev/null || echo "${OUT}/${PREFIX}_tpid_sol.bin")" \
  "${OUT}/${PREFIX}_tpid_sol.bin" 2>/dev/null || true
# ensure sol present in OUT
if [[ ! -e "${OUT}/${PREFIX}_tpid_sol.bin" ]]; then
  echo "missing tpid sol ${PREFIX}_tpid_sol.bin"; ls -la "${OUT}"; exit 1
fi

cd "${OUT}"
START=$(date +%s)
mpirun -np 4 env DENDRO_RHS_PT_ASYNC=1 DENDRO_DUMP_BLK_LEV=1 \
  "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
END=$(date +%s)
echo "wall_sec $((END-START))" | tee -a "${LOG}"
grep -E 'BLK_LEV|C_HAM|ETS time|Grid Info|wall_sec|level' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | head -80
echo Q4_SHORT_4GPU_DONE
