#!/usr/bin/env bash
# B3 pack A/B on 4 GPUs (serial-K ghosts vs ensK packed).
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
LOG="${HOME}/Dendro-GR/campaign-notes/tier0/b3_pack_bench_4gpu.log"
echo "host $(hostname) $(date -Is) b3_pack_4gpu" | tee "${LOG}"

echo "==== build bssnSolverCUDA ====" | tee -a "${LOG}"
cmake --build "${DENDRO_BUILD}" -j16 --target bssnSolverCUDA 2>&1 | tee -a "${LOG}" | tail -20

SOLBIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q8.meshprobe.par.toml"
SOL="${HOME}/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/b3_pack_bench_4gpu"
PREFIX=$(grep '^TPID_FILEPREFIX' "${PAR0}" | sed 's/.*= *"\([^"]*\)".*/\1/')
rm -rf "${OUT}"; mkdir -p "${OUT}"/{aeh,cp,dat,vtu}
cp "${PAR0}" "${OUT}/local.par.toml"
python3 - <<PY
from pathlib import Path
import re
p=Path("${OUT}/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.05", t)
t=re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t=re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = 0", t)
p.write_text(t)
PY
ln -sfn "${SOL}" "${OUT}/${PREFIX}_tpid_sol.bin"
cd "${OUT}"

for K in 2 4 8; do
  echo "==== B3 pack bench K=$K np=4 ====" | tee -a "${LOG}"
  START=$(date +%s)
  # argv[2] is ts_mode (1=ETS), not npes
  mpirun -np 4 env -u CUDA_VISIBLE_DEVICES \
    DENDRO_RHS_PT_ASYNC=1 DENDRO_B3_PACK_BENCH=$K \
    "${SOLBIN}" local.par.toml 1 >"run_K${K}.log" 2>&1 || true
  END=$(date +%s)
  grep -E 'b3_pack_bench|ETS time' "run_K${K}.log" | sed 's/\x1b\[[0-9;]*m//g' | tee -a "${LOG}" || true
  echo "K=$K wall=$((END-START))s" | tee -a "${LOG}"
done
echo B3_PACK_BENCH_DONE | tee -a "${LOG}"
