#!/usr/bin/env bash
# Resume Tier 2.5 for np=2 and np=4 (np=1 already measured).
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile 2>/dev/null || true
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
BASE="${HOME}/Dendro-GR-runs/q8_scale"
LOGDIR="${HOME}/Dendro-GR/campaign-notes/tier0"
SUMMARY="${LOGDIR}/tier25_scale_q8.log"
PREFIX=mesh_q8
SOL="${BASE}/tpid/${PREFIX}_tpid_sol.bin"

for NP in 2 4; do
  OUT="${BASE}/np${NP}"
  LOG="${LOGDIR}/tier25_q8_np${NP}.log"
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
  ln -sfn "${SOL}" "${OUT}/${PREFIX}_tpid_sol.bin"
  cd "${OUT}"
  echo "==== np=${NP} ====" | tee -a "${SUMMARY}"
  START=$(date +%s)
  mpirun -np "${NP}" env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 DENDRO_DUMP_BLK_LEV=1 \
    "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
  END=$(date +%s)
  WALL=$((END-START))
  ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
  echo "np=${NP} wall_sec=${WALL} ets=${ETS:-NA}" | tee -a "${SUMMARY}"
  grep -E 'BLK_LEV_HIST|ETS time|npes' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -10 | tee -a "${SUMMARY}" || true
done

python3 - <<'PY' | tee -a /home1/08435/srvenkat/Dendro-GR/campaign-notes/tier0/tier25_scale_q8.log
import re
from pathlib import Path
text=Path('/home1/08435/srvenkat/Dendro-GR/campaign-notes/tier0/tier25_scale_q8.log').read_text()
rows={}
for m in re.finditer(r'np=(\d+) wall_sec=(\d+|NA) ets=([0-9.eE+-]+|NA)', text):
    np=int(m.group(1)); wall=m.group(2); ets=m.group(3)
    rows[np]=(wall, None if ets=='NA' else float(ets))
print('np\twall\tETS\tspeedup_ets\teff_ets')
t1=rows.get(1,(None,None))[1]
for np in sorted(rows):
    wall, ets = rows[np]
    if ets is None or t1 is None:
        print(f'{np}\t{wall}\t{ets}\tNA\tNA'); continue
    sp=t1/ets; eff=sp/np
    print(f'{np}\t{wall}\t{ets:.4f}\t{sp:.3f}\t{eff:.3f}')
print('TIER25_SCALE_Q8_DONE')
PY
