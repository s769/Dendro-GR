#!/usr/bin/env bash
# Dual-res FD v2 (after init-mesh replay fix): record BOTH fiducials, replay m0/p/m on each,
# and self-check record-vs-replay C_HAM at m0 (must agree ~1e-15 rel).
set -euo pipefail
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" OMP_NUM_THREADS=1
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier15_dualres_v2"; mkdir -p "${OUT}"

run() {
  local tag="$1" par="$2" mass="$3" mode="$4" meshdir="$5"
  local d="${OUT}/${tag}"; mkdir -p "${d}"/{aeh,cp,dat,vtu}
  cp "${par}" "${d}/local.par.toml"
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
  if [[ "$mode" == record ]]; then
    mpirun -np 1 env DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_RECORD="${meshdir}" "${BIN}" "${d}/local.par.toml" 1 >run.log 2>&1
  else
    mpirun -np 1 env DENDRO_RHS_PT_ASYNC=1 DENDRO_MESH_REPLAY="${meshdir}" "${BIN}" "${d}/local.par.toml" 1 >run.log 2>&1
  fi
  python3 - <<PY
from pathlib import Path
rows=[]
for i,l in enumerate(Path("dat/bssn_profile_Constraints.dat").read_text().splitlines()):
    if not l.strip() or l.startswith("TimeStep") or "C_HAM" in l.split()[:3]:
        continue
    parts=l.split()
    try:
        rows.append(float(parts[2]))
    except (ValueError, IndexError):
        continue
print("${tag}", rows[-1], "n", len(rows))
Path("ham.txt").write_text(str(rows[-1]))
PY
}

COARSE_REC="${OUT}/mesh_rec_coarse"
FINE_PAR="${HOME}/Dendro-GR-runs/q1.smoke_hfine.par.toml"
FINE_REC="${OUT}/mesh_rec_fine"

# reuse recorded meshes from v2; scan eps
M0=0.4824
for res in coarse fine; do
  par="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"; rec="${COARSE_REC}"
  if [[ "$res" == fine ]]; then par="${FINE_PAR}"; rec="${FINE_REC}"; fi
  for EPS in 0.0003 0.003 0.01; do
    run "${res}_p_${EPS}" "${par}" "$(python3 -c "print(${M0}+${EPS})")" replay "${rec}"
    run "${res}_m_${EPS}" "${par}" "$(python3 -c "print(${M0}-${EPS})")" replay "${rec}"
  done
done
python3 - <<'PY'
from pathlib import Path
O=Path.home()/'Dendro-GR-runs/tier15_dualres_v2'
for r in ['coarse','fine']:
    for e in ['0.0003','0.001','0.003','0.01']:
        sp,sm=('p','m') if e=='0.001' else (f'p_{e}',f'm_{e}')
        hp=float((O/f'{r}_{sp}/ham.txt').read_text()); hm=float((O/f'{r}_{sm}/ham.txt').read_text())
        print(r,e,(hp-hm)/(2*float(e)))
print('SCAN_DONE')
PY
