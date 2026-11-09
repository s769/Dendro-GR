#!/usr/bin/env bash
# Tier 2.5 redo: q8 strong scale with -DMPIX_CUDA_AWARE_SUPPORT=ON binary.
set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile 2>/dev/null || true
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export UCX_TLS="${UCX_TLS:-rc,sm,cuda_copy,cuda_ipc,gdr_copy}"

BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
BASE="${HOME}/Dendro-GR-runs/q8_scale_cudaaware"
LOGDIR="${HOME}/Dendro-GR-notes/tier0"
SUMMARY="${LOGDIR}/tier25_scale_q8_cudaaware.log"
mkdir -p "${BASE}" "${LOGDIR}"
: > "${SUMMARY}"
echo "host $(hostname) $(date -Is) CUDA_AWARE=ON" | tee -a "${SUMMARY}"

for NP in 1 2 4; do
  OUT="${BASE}/np${NP}"
  LOG="${LOGDIR}/tier25_q8_cudaaware_np${NP}.log"
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
  echo "==== np=${NP} ====" | tee -a "${SUMMARY}"
  START=$(date +%s)
  mpirun -np "${NP}" env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
    "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
  END=$(date +%s)
  WALL=$((END-START))
  ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
  echo "np=${NP} wall_sec=${WALL} ets=${ETS:-NA}" | tee -a "${SUMMARY}"
done

python3 - <<'PY' | tee -a /home1/08435/srvenkat/Dendro-GR-notes/tier0/tier25_scale_q8_cudaaware.log
import re
from pathlib import Path
text=Path('/home1/08435/srvenkat/Dendro-GR-notes/tier0/tier25_scale_q8_cudaaware.log').read_text()
rows={}
for m in re.finditer(r'np=(\d+) wall_sec=(\d+|NA) ets=([0-9.eE+-]+|NA)', text):
    rows[int(m.group(1))]=(m.group(2), None if m.group(3)=='NA' else float(m.group(3)))
print('np\twall\tETS\tspeedup\teff')
t1=rows.get(1,(None,None))[1]
for np in sorted(rows):
    wall, ets = rows[np]
    if ets is None or t1 is None:
        print(f'{np}\t{wall}\t{ets}\tNA\tNA'); continue
    print(f'{np}\t{wall}\t{ets:.4f}\t{t1/ets:.3f}\t{(t1/ets)/np:.3f}')
print('TIER25_SCALE_Q8_CUDAAWARE_DONE')
PY
