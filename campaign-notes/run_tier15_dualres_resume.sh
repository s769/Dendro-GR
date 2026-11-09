#!/usr/bin/env bash
# Resume dual-res FD after interrupt: finish missing fine_{p,m} then print FD table.
set -euo pipefail
source "${HOME}/Dendro-GR-notes/env_build_horizon.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier15_dualres"
FINE_PAR="${HOME}/Dendro-GR-runs/q1.smoke_hfine.par.toml"
FINE_REC="${OUT}/mesh_rec_fine"
M0=0.4824; EPS=0.001
mkdir -p "${OUT}"

run_replay() {
  local tag="$1" mass="$2"
  local d="${OUT}/${tag}"
  if [[ -f "${d}/ham.txt" ]]; then
    echo "skip ${tag} (ham.txt exists: $(cat "${d}/ham.txt"))"
    return 0
  fi
  mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${FINE_PAR}" "${d}/local.par.toml"
  python3 - <<PY
from pathlib import Path
import re
p=Path("${d}/local.par.toml"); t=p.read_text()
t=re.sub(r"(BSSN_BH1 = \{ MASS = )[0-9.eE+-]+", r"\g<1>${mass}", t, count=1)
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = 1.0", t)
p.write_text(t)
PY
  ln -sfn "${TPID}" "${d}/rit_q1_tpid_sol.bin"
  cd "${d}"
  env DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_REPLAY="${FINE_REC}" "${BIN}" "${d}/local.par.toml" 1 >run.log 2>&1
  python3 - <<PY
from pathlib import Path
rows=[]
for l in Path("dat/bssn_profile_Constraints.dat").read_text().splitlines():
    if not l.strip() or l.startswith("TimeStep") or "C_HAM" in l.split()[:3]:
        continue
    parts=l.split()
    try:
        rows.append(float(parts[2]))
    except (ValueError, IndexError):
        continue
assert rows, "no C_HAM rows in Constraints.dat"
print("${tag}", rows[-1], "n", len(rows))
Path("ham.txt").write_text(str(rows[-1]))
PY
}

run_replay fine_p "$(python3 -c "print(${M0}+${EPS})")"
run_replay fine_m "$(python3 -c "print(${M0}-${EPS})")"

python3 - <<'PY'
from pathlib import Path
OUT=Path('/home1/08435/srvenkat/Dendro-GR-runs/tier15_dualres')
eps=1e-3
print('res\tFD_central_C_HAM')
for res in ['coarse','fine']:
    hp=float((OUT/f'{res}_p/ham.txt').read_text())
    hm=float((OUT/f'{res}_m/ham.txt').read_text())
    print(f'{res}\t{(hp-hm)/(2*eps):.8e}')
print('TIER15_DUALRES_DONE')
PY
