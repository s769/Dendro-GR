#!/usr/bin/env bash
# B1 while-hot RIJ A/B vs PT_ASYNC.
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
LOG="${HOME}/Dendro-GR-notes/tier0/tier21_whilehot_ab.log"
echo "host $(hostname) $(date -Is) whilehot_ab" | tee "${LOG}"

echo "==== build rhsgpu_test ====" | tee -a "${LOG}"
cmake --build "${DENDRO_BUILD}" -j16 --target rhsgpu_test 2>&1 | tee -a "${LOG}" | tail -25

BIN="${DENDRO_BUILD}/BSSN_GR/rhsgpu_test"
echo "==== rhsgpu 256 bit-id+time ====" | tee -a "${LOG}"
DENDRO_TIME_PT_ASYNC=1 DENDRO_TIME_PT_ASYNC_WHILEHOT=1 \
  "${BIN}" 256 25 0 2>&1 | tee -a "${LOG}" | grep -E 'pt_async_(gpu|whilehot)' || true

echo "==== rhsgpu 2048 ====" | tee -a "${LOG}"
DENDRO_TIME_PT_ASYNC=1 DENDRO_TIME_PT_ASYNC_WHILEHOT=1 \
  "${BIN}" 2048 20 0 2>&1 | tee -a "${LOG}" | grep -E 'pt_async_(gpu|whilehot)' || true

echo "==== build bssnSolverCUDA ====" | tee -a "${LOG}"
cmake --build "${DENDRO_BUILD}" -j16 --target bssnSolverCUDA 2>&1 | tee -a "${LOG}" | tail -15

SOLBIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier21_whilehot_ab"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
rm -rf "${OUT}"; mkdir -p "${OUT}"/{ASYNC,WHILEHOT}/{aeh,cp,dat,vtu}
for tag in ASYNC WHILEHOT; do
  cp "${PAR0}" "${OUT}/${tag}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/${tag}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.25", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 0", t)
p.write_text(t)
PY
  ln -sfn "${SOL}" "${OUT}/${tag}/${PREFIX}_tpid_sol.bin"
done

run_one() {
  local tag="$1" envflag="$2"
  cd "${OUT}/${tag}"
  START=$(date +%s)
  mpirun -np 1 env -u CUDA_VISIBLE_DEVICES ${envflag}=1 \
    "${SOLBIN}" local.par.toml 1 >run.log 2>&1
  END=$(date +%s)
  ETS=$(grep -E 'ETS time \(max\)' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
  echo "${tag} wall=$((END-START))s ets=${ETS:-NA}" | tee -a "${LOG}"
}
run_one ASYNC DENDRO_RHS_PT_ASYNC
run_one WHILEHOT DENDRO_RHS_PT_ASYNC_WHILEHOT
echo WHILEHOT_AB_DONE | tee -a "${LOG}"
