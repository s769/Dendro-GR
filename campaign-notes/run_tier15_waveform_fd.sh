#!/usr/bin/env bash
# Tier 1.5 waveform FD: d(psi4_22 phase/amp)/d(BH1.MASS) vs eps, frozen mesh, one resolution per call.
# Usage: RES=coarse|fine TEND=60 RFREQ=20 RADII="[20.0,25.0,30.0,35.0,40.0,45.0]" EPSS="0.001 0.003" bash this
# Needs 4 GPUs on one node (fiducial first, then p/m for 2 eps in parallel).
set -uo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "${HOME}/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null
unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=1
RES="${RES:-coarse}"; TEND="${TEND:-60}"; RFREQ="${RFREQ:-20}"
RADII="${RADII:-[20.0, 25.0, 30.0, 35.0, 40.0, 45.0]}"; EPSS="${EPSS:-0.001 0.003}"
BIN="${DENDRO_BUILD}/BSSN_GR/bssnSolverCUDA"
PAR0="${HOME}/Dendro-GR-runs/q1.smoke_cons.par.toml"; [[ "$RES" == fine ]] && PAR0="${HOME}/Dendro-GR-runs/q1.smoke_hfine.par.toml"
TPID="${HOME}/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin"
OUT="${HOME}/Dendro-GR-runs/tier15_wave_${RES}_T${TEND}${OUTTAG:-}"; mkdir -p "${OUT}"; REC="${OUT}/mesh_rec"; mkdir -p "$REC"
M0=0.4824

setup() { # tag mass
  local d="${OUT}/$1"; mkdir -p "$d"/{aeh,cp,dat,vtu}; cp "$PAR0" "$d/local.par.toml"
  python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
t=re.sub(r"(BSSN_BH1 = \{ MASS = )[0-9.eE+-]+", r"\g<1>$2", t, count=1)
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+", "BSSN_RK_TIME_END = %s" % float("${TEND}"), t)
t=re.sub(r"BSSN_REMESH_TEST_FREQ = [0-9]+", "BSSN_REMESH_TEST_FREQ = ${RFREQ}", t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+", "BSSN_GW_EXTRACT_FREQ = ${GWFREQ:-4}", t)
t=re.sub(r"BSSN_GW_RADAII = \[[^\]]*\]", "BSSN_GW_RADAII = ${RADII}", t)
p.write_text(t)
PY
  ln -sfn "$TPID" "$d/rit_q1_tpid_sol.bin"
}
runit() { # tag gpu mode
  local d="${OUT}/$1"
  ( cd "$d"; local s=$(date +%s)
    if [[ "$3" == record ]]; then E="DENDRO_MESH_RECORD=${REC}"; else E="DENDRO_MESH_REPLAY=${REC}"; fi
    CUDA_VISIBLE_DEVICES=$2 mpirun -np 1 env DENDRO_RHS_PT_ASYNC=1 $E "$BIN" local.par.toml 1 > run.log 2>&1
    echo "$1 gpu$2 wall=$(( $(date +%s)-s ))s $(grep -E 'ETS time' run.log | tail -1)" ) 
}
echo "RES=$RES TEND=$TEND RFREQ=$RFREQ RADII=$RADII EPSS=$EPSS $(date -Is) $(hostname)"
setup fid "$M0"; runit fid 0 record; [[ -f "${OUT}/fid/dat/bssn_profile_GW_l2_m2.dat" ]] || { echo FID_FAILED; exit 1; }
i=0
for e in $EPSS; do
  mp=$(python3 -c "print($M0+$e)"); mm=$(python3 -c "print($M0-$e)")
  setup "p_$e" "$mp"; setup "m_$e" "$mm"
  runit "p_$e" $((i%4)) replay & i=$((i+1))
  runit "m_$e" $((i%4)) replay & i=$((i+1))
done
wait
python3 "${HOME}/Dendro-GR/campaign-notes/analyze_wave_fd.py" "$OUT" $EPSS
echo WAVE_FD_DONE
