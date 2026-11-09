#!/usr/bin/env bash
# Tier 2.5: q=8 strong scaling on one node (1/2/4 GPUs), short evolve.
set -euo pipefail
# Non-interactive compute-node shells need explicit lmod init.
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile 2>/dev/null || true
  source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
fi
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
# env_build pins CUDA_VISIBLE_DEVICES=0 — clear so ranks can see all GPUs.
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
TPID_BIN="${DENDRO_BUILD}/BSSN_GR/tpid"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
BASE="${HOME}/Dendro-GR-runs/q8_scale"
LOGDIR="${HOME}/Dendro-GR-notes/tier0"
SUMMARY="${LOGDIR}/tier25_scale_q8.log"
mkdir -p "${BASE}" "${LOGDIR}"
: > "${SUMMARY}"

PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
# Shared tpid sol under BASE (generate once with 1 rank).
SOL_DIR="${BASE}/tpid"
mkdir -p "${SOL_DIR}"
HIST_SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/${PREFIX}_tpid_sol.bin"
if [[ ! -f "${SOL_DIR}/${PREFIX}_tpid_sol.bin" ]]; then
  if [[ -f "${HIST_SOL}" ]]; then
    ln -sfn "${HIST_SOL}" "${SOL_DIR}/${PREFIX}_tpid_sol.bin"
  else
    cp "${PAR0}" "${SOL_DIR}/local.par.toml"
    (cd "${SOL_DIR}" && "${TPID_BIN}" local.par.toml 1 > tpid.log 2>&1)
  fi
fi
if [[ ! -f "${SOL_DIR}/${PREFIX}_tpid_sol.bin" ]]; then
  echo "NO_Q8_TPID"; ls -la "${SOL_DIR}"; exit 1
fi

echo "host $(hostname)  $(date -Is)" | tee -a "${SUMMARY}"
nvidia-smi -L 2>/dev/null | tee -a "${SUMMARY}" || true

for NP in 1 2 4; do
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
  ln -sfn "${SOL_DIR}/${PREFIX}_tpid_sol.bin" "${OUT}/${PREFIX}_tpid_sol.bin"
  cd "${OUT}"
  echo "==== np=${NP} ====" | tee -a "${SUMMARY}"
  START=$(date +%s)
  # Leave all GPUs visible; cuda_init maps MPI rank → device.
  mpirun -np "${NP}" \
    env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 DENDRO_DUMP_BLK_LEV=1 \
    "${BIN}" "${OUT}/local.par.toml" 1 > "${LOG}" 2>&1 || true
  END=$(date +%s)
  WALL=$((END-START))
  ETS=$(grep -E 'ETS time \(max\)' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}' || true)
  HAM=$(grep -E 'Constraint C_HAM L2' "${LOG}" | sed 's/\x1b\[[0-9;]*m//g' | tail -1 || true)
  echo "np=${NP} wall_sec=${WALL} ets=${ETS:-NA}" | tee -a "${SUMMARY}"
  echo "  ${HAM:-no_ham}" | tee -a "${SUMMARY}"
  grep -E 'BLK_LEV_HIST|Grid Info|number of octs|nBlocks|wall_sec|ETS time' "${LOG}" \
    | sed 's/\x1b\[[0-9;]*m//g' | tail -20 | tee -a "${SUMMARY}" || true
done

# Parallel efficiency vs 1 GPU (ETS).
python3 - <<'PY' | tee -a /home1/08435/srvenkat/Dendro-GR-notes/tier0/tier25_scale_q8.log
import re
from pathlib import Path
text=Path('/home1/08435/srvenkat/Dendro-GR-notes/tier0/tier25_scale_q8.log').read_text()
rows={}
for m in re.finditer(r'np=(\d+) wall_sec=(\d+) ets=([0-9.eE+-]+|NA)', text):
    np, wall, ets = int(m.group(1)), int(m.group(2)), m.group(3)
    rows[np]=(wall, None if ets=='NA' else float(ets))
print('np\twall\tETS\tspeedup_ets\teff_ets')
if 1 in rows and rows[1][1]:
    t1=rows[1][1]
    for np in sorted(rows):
        wall, ets = rows[np]
        if ets is None:
            print(f'{np}\t{wall}\tNA\tNA\tNA'); continue
        sp=t1/ets; eff=sp/np
        print(f'{np}\t{wall}\t{ets:.4f}\t{sp:.3f}\t{eff:.3f}')
print('TIER25_SCALE_Q8_DONE')
PY
