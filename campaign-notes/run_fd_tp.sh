#!/usr/bin/env bash
# Production-style FD (BSSN_ID_TYPE=0 TwoPunctures): fiducial (record) + members with their OWN warm-started TP solves (BH1 bare mass perturbed).
# q1.par.toml adaptivity, cap CAP, T=TEND.  Members: +-EPS1, +-EPS2 (bare mass of BH1 = 0.5 +- eps).
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
CAP=${CAP:-12}; TEND=${TEND:-110}; EPS="${EPS:-0.001 0.003}"; M0=0.5
OUT=$HOME/Dendro-GR-runs/fdtp_c${CAP}; mkdir -p $OUT; REC=$OUT/mesh_rec; mkdir -p $REC
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA; TPID=$DENDRO_BUILD/BSSN_GR/tpid
TPB=$HOME/Dendro-GR-runs/resstudy2/tp/tp_q001_tpid_sol.bin           # base solution (BH1=BH2=0.5 bare)
gwf=$(( 8 * (1 << (CAP - 11)) ))
setup() { # tag mass
  local d=$OUT/$1; rm -rf $d; mkdir -p $d/{aeh,bah,cp,dat,vtu}; cp $HOME/Dendro-GR/BSSN_GR/pars/q1.par.toml $d/local.par.toml
  python3 - <<PY
import re
from pathlib import Path
p=Path("$d/local.par.toml"); t=p.read_text()
t=re.sub(r"(BSSN_BH1 = \{ MASS = )[0-9.eE+-]+", r"\g<1>$2", t, count=1)
def sub(k,v):
    global t
    t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M)
    if n==0:
        m=re.search(r"^\[",t,flags=re.M); ins=k+" = "+v+"\n\n"; t=(t[:m.start()]+ins+t[m.start():]) if m else t+"\n"+ins
for k,v in [("BSSN_RESTORE_SOLVER","0"),("BSSN_MAXDEPTH","$CAP"),("BSSN_BH1_MAX_LEV","$CAP"),("BSSN_BH2_MAX_LEV","$CAP"),("BSSN_RK_TIME_END","$TEND.0"),
            ("BSSN_GW_EXTRACT_FREQ","$gwf"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0")]: sub(k,v)
p.write_text(t)
PY
}
run() { # tag gpu mode(record|replay)
  ( cd $OUT/$1; s=$(date +%s)
    if [ "$3" = record ]; then E="DENDRO_MESH_RECORD=$REC"; else E="DENDRO_MESH_REPLAY=$REC"; fi
    CUDA_VISIBLE_DEVICES=$2 mpirun -np 1 --bind-to none env OMP_NUM_THREADS=${OMPT:-8} DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1 $E $BIN local.par.toml 1 > run.log 2>&1
    echo "$1 wall=$(( $(date +%s)-s ))s $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1)" >> $OUT/summary.txt )
}
if [ -z "${REPLAY_ONLY:-}" ]; then
# fiducial: base TP solution, record the mesh sequence
setup fid $M0; ln -sfn $TPB $OUT/fid/tp_q001_tpid_sol.bin; run fid 0 record &
FID=$!
# members: own warm-started TP solves (36 threads, sequential) while the fiducial runs
for e in $EPS; do
  for sgn in p m; do
    mass=$(python3 -c "print($M0 $( [ $sgn = p ] && echo + || echo - ) $e)")
    setup ${sgn}_$e $mass; d=$OUT/${sgn}_$e
    ( cd $d; env OMP_NUM_THREADS=36 TP_OMP_THREADS=36 DENDRO_TPID_WARM_START=$TPB $TPID local.par.toml 36 > tpid.log 2>&1 ; echo "${sgn}_$e TP: $(grep 'TP solver time' tpid.log | tail -1) | $(grep -c 'Newton: it=' tpid.log) Newton lines | last $(grep 'Newton: it=' tpid.log | tail -1)" >> $OUT/summary.txt )
    [ -f $d/tp_q001_tpid_sol.bin ] || { echo "TP SOLVE FAILED for ${sgn}_$e" >> $OUT/summary.txt; }
  done
done
wait $FID
fi
i=0
for e in $EPS; do for sgn in p m; do run ${sgn}_$e $((i%4)) replay & i=$((i+1)); done; done
wait
echo FDTP_DONE >> $OUT/summary.txt
