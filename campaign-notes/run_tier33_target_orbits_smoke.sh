#!/usr/bin/env bash
# Tier 3.3 scaffold smoke: BSSN_TARGET_ORBITS prints PN sep and updates BH X / par_b.
# Uses existing TPID sol (ID inconsistent by design for this knob check only).
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source /home1/08435/srvenkat/Dendro-GR/campaign-notes/env_build_horizon.sh
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

LOG="${HOME}/Dendro-GR/campaign-notes/tier0/tier33_target_orbits_smoke.log"
OUT="${HOME}/Dendro-GR-runs/tier33_target_orbits_smoke"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
N_ORBITS="${1:-5.0}"
# toml11 as_floating() rejects bare integers — always write a float literal.
N_ORBITS_F=$(python3 -c "print(float('${N_ORBITS}'))")

mkdir -p "${OUT}"/{aeh,cp,dat,vtu} "$(dirname "${LOG}")"
echo "host $(hostname) $(date -Is) TARGET_ORBITS=${N_ORBITS_F}" | tee "${LOG}"

# Rebuild solver (parameters.cpp change) unless SKIP_BUILD=1
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build "${DENDRO_BUILD}" -j16 --target bssnSolverCUDA 2>&1 | tee -a "${LOG}" | tail -20
fi

cp "${PAR0}" "${OUT}/local.par.toml"
python3 - <<PY
from pathlib import Path
import re
p = Path("${OUT}/local.par.toml")
t = p.read_text()
t = re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 0.05", t)
t = re.sub(r"BSSN_IO_OUTPUT_FREQ = [0-9]+", "BSSN_IO_OUTPUT_FREQ = 0", t)
t = re.sub(r"BSSN_CHECKPT_FREQ = [0-9]+", "BSSN_CHECKPT_FREQ = 0", t)
n = float("${N_ORBITS_F}")
line = f"BSSN_TARGET_ORBITS = {n}"
if "BSSN_TARGET_ORBITS" in t:
    t = re.sub(r"BSSN_TARGET_ORBITS = [0-9.eE+-]+", line, t)
else:
    t = line + "\n" + t
p.write_text(t)
print("wrote", p, line)
PY

# Expected PN: equal-mass nu=1/4, a/M=(32*pi*nu*N)^(2/5), b=a/2
python3 - <<PY | tee -a "${LOG}"
import math
N=float("${N_ORBITS}"); nu=0.25; M=0.9648
aM=(32*math.pi*nu*N)**(2/5); b=0.5*aM*M
print(f"expect N={N} a/M={aM:.6f} par_b={b:.6f}")
PY

cd "${OUT}"
# Link TPID sol if present next to original
PREFIX=$(grep '^TPID_FILEPREFIX' local.par.toml | sed 's/.*= *"\([^"]*\)".*/\1/')
SOL_LINKED=
for cand in \
  "${HOME}/Dendro-GR-runs/smoke_cons_pt/${PREFIX}_tpid_sol.bin" \
  "${HOME}/Dendro-GR-runs/mesh_rec_fiducial/${PREFIX}_tpid_sol.bin" \
  "${HOME}/Dendro-GR-runs/b3_pack_q1_4gpu/${PREFIX}_tpid_sol.bin" \
  "${HOME}/Dendro-GR-runs/${PREFIX}_tpid_sol.bin" \
  "$(dirname "${PAR0}")/${PREFIX}_tpid_sol.bin"; do
  if [[ -f "$cand" ]]; then
    ln -sfn "$cand" "${OUT}/${PREFIX}_tpid_sol.bin"
    echo "linked TPID sol $cand" | tee -a "${LOG}"
    SOL_LINKED=1
    break
  fi
done
[[ -n "${SOL_LINKED}" ]] || echo "WARN: no TPID sol found for ${PREFIX}" | tee -a "${LOG}"

START=$(date +%s)
mpirun -np 1 env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 \
  "${BIN}" local.par.toml 1 >run.log 2>&1 || {
  echo FAIL_T33_SMOKE | tee -a "${LOG}"
  tail -50 run.log | tee -a "${LOG}"
  exit 1
}
END=$(date +%s)
grep -E 'TARGET_ORBITS|TPID_PAR_B|par_b|ETS time' run.log | sed 's/\x1b\[[0-9;]*m//g' | tee -a "${LOG}" || true
echo "wall=$((END-START))s" | tee -a "${LOG}"
if grep -q '\[TARGET_ORBITS\]' run.log; then
  echo T33_TARGET_ORBITS_SMOKE_PASS | tee -a "${LOG}"
else
  echo T33_TARGET_ORBITS_SMOKE_FAIL_NO_PRINT | tee -a "${LOG}"
  exit 1
fi
