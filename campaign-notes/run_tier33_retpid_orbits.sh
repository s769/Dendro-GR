#!/usr/bin/env bash
# Tier 3.3: re-solve TPID for each BSSN_TARGET_ORBITS N (q1 equal-mass).
set -euo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1

TPID="${DENDRO_BUILD}/BSSN_GR/tpid"
PAR0="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"
OUT="${HOME}/Dendro-GR-runs/tpid_tier33_orbits"
LOG="${HOME}/Dendro-GR-notes/tier0/tier33_retpid_orbits.log"
NS="${*:-3.0 5.0 8.0}"

mkdir -p "${OUT}" "$(dirname "${LOG}")"
echo "host $(hostname) $(date -Is) retpid orbits: ${NS}" | tee "${LOG}"
test -x "${TPID}" || { echo "missing tpid binary"; exit 1; }

for N in ${NS}; do
  Nf=$(python3 -c "print(float('${N}'))")
  tag=$(python3 -c "print(f'N{float(\"${N}\"):g}'.replace('.','p'))")
  d="${OUT}/${tag}"
  mkdir -p "${d}"
  cp "${PAR0}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p = Path("${d}/local.par.toml")
t = p.read_text()
line = f"BSSN_TARGET_ORBITS = ${Nf}"
if "BSSN_TARGET_ORBITS" in t:
    t = re.sub(r"BSSN_TARGET_ORBITS = [0-9.eE+-]+", line, t)
else:
    t = line + "\n" + t
t = re.sub(r'TPID_FILEPREFIX = "[^"]*"', f'TPID_FILEPREFIX = "rit_q1_${tag}"', t)
p.write_text(t)
print("wrote", p, line)
PY
  echo "==== ${tag} N=${Nf} ====" | tee -a "${LOG}"
  START=$(date +%s)
  ( cd "${d}" && "${TPID}" local.par.toml 4 >tpid.log 2>&1 ) || {
    echo FAIL_TPID "${tag}" | tee -a "${LOG}"
    tail -40 "${d}/tpid.log" | tee -a "${LOG}"
    exit 1
  }
  END=$(date +%s)
  sol="${d}/rit_q1_${tag}_tpid_sol.bin"
  test -f "${sol}"
  grep -E 'TARGET_ORBITS|TPID_PAR_B|par_b|two puncture|resolved inversion' "${d}/tpid.log" \
    | sed 's/\x1b\[[0-9;]*m//g' | tee -a "${LOG}" || true
  echo "${tag} wall=$((END-START))s sol=$(du -h "${sol}" | awk '{print $1}')" | tee -a "${LOG}"
done
echo T33_RETPID_ORBITS_DONE | tee -a "${LOG}"
