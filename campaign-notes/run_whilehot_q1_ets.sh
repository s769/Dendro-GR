#!/usr/bin/env bash
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source /home1/08435/srvenkat/Dendro-GR/campaign-notes/env_build_horizon.sh
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="/home1/08435/srvenkat/Dendro-GR-runs/q1.smoke_cons.par.toml"
OUT="/home1/08435/srvenkat/Dendro-GR-runs/whilehot_q1_ets"
LOG="/home1/08435/srvenkat/Dendro-GR/campaign-notes/tier0/whilehot_fine_ensk.log"
mkdir -p "${OUT}/WHILEHOT"/{aeh,cp,dat,vtu}
cp "${PAR0}" "${OUT}/WHILEHOT/local.par.toml"
cd "${OUT}/WHILEHOT"
echo "host $(hostname) $(date -Is) q1_WHILEHOT" | tee -a "${LOG}"
echo "==== q1 WHILEHOT ====" | tee -a "${LOG}"
START=$(date +%s)
mpirun -np 1 --bind-to none env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 DENDRO_RHS_PT_ASYNC_WHILEHOT=1 \
  "${BIN}" local.par.toml 1 >run.log 2>&1 || {
  echo FAIL_WHILEHOT_q1 | tee -a "${LOG}"
  tail -40 run.log | tee -a "${LOG}"
  exit 1
}
END=$(date +%s)
ETS=$(grep -E 'ETS time \(max\)' run.log | sed 's/\x1b\[[0-9;]*m//g' | tail -1 | awk '{print $NF}')
echo "q1 WHILEHOT wall=$((END-START))s ets=${ETS:-NA}" | tee -a "${LOG}"
python3 - <<PY | tee -a "${LOG}"
a=12.4584
w=float("${ETS}" or "nan")
print(f"q1_whilehot_vs_async_delta_ets {w-a:+.4f} rel {(w-a)/a*100:+.2f}%")
print("WHILEHOT_FINE_ENSK_DONE")
PY
