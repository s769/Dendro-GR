#!/usr/bin/env bash
# Production-like accuracy-vs-cost study (q1.par.toml adaptivity: wavelet-tol function 6, ramp r=3..12, AMR_FAC 0.2, GW radii 50..100)
# usage: TEND=110 run_resstudy2.sh "name:cap:tol:gpu ..."   (one single-rank process per GPU, concurrent)
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
TEND=${TEND:-110}; OUT=$HOME/Dendro-GR-runs/resstudy2; mkdir -p $OUT
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
# one shared TwoPunctures solve for the production q1 parameters (solver needs a pre-solved file; identical IC for every variant)
TPD=$OUT/tp; mkdir -p $TPD
if [ ! -f $TPD/tp_q001_tpid_sol.bin ]; then
  cp $HOME/Dendro-GR/BSSN_GR/pars/q1.par.toml $TPD/q1.par.toml; ( cd $TPD; mkdir -p aeh bah cp dat vtu; env OMP_NUM_THREADS=36 TP_OMP_THREADS=36 $DENDRO_BUILD/BSSN_GR/tpid q1.par.toml 36 > tpid.log 2>&1 )
fi
[ -f $TPD/tp_q001_tpid_sol.bin ] || { echo "TPID solve failed (see $TPD/tpid.log)"; exit 1; }
for spec in $1; do
  IFS=: read name cap tol gpu extra <<< "$spec"      # extra = "KEY=VAL;KEY=VAL" extra par overrides (optional)
  gwf=$(( 8 * (1 << (cap - 11)) ))        # constant sample interval dT = 0.5208 M for every cap
  d=$OUT/${name}_T${TEND}; rm -rf $d; mkdir -p $d/{aeh,bah,cp,dat,vtu}; cp $HOME/Dendro-GR/BSSN_GR/pars/q1.par.toml $d/local.par.toml
  ln -sfn $TPD/tp_q001_tpid_sol.bin $d/tp_q001_tpid_sol.bin
  python3 - <<PY
import re
from pathlib import Path
p=Path("$d/local.par.toml"); t=p.read_text()
def sub(k,v):
    global t
    t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M)
    if n==0:
        m=re.search(r"^\[",t,flags=re.M); ins=k+" = "+v+"\n\n"
        t=(t[:m.start()]+ins+t[m.start():]) if m else t+"\n"+ins
for k,v in [("BSSN_RESTORE_SOLVER","0"),("BSSN_MAXDEPTH","$cap"),("BSSN_BH1_MAX_LEV","$cap"),("BSSN_BH2_MAX_LEV","$cap"),
            ("BSSN_WAVELET_TOL","$tol"),("BSSN_RK_TIME_END","$TEND.0"),("BSSN_GW_EXTRACT_FREQ","$gwf"),
            ("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0")]+[tuple(x.split("=",1)) for x in "$extra".split(";") if x]: sub(k,v)
p.write_text(t)
PY
  ( cd $d; s=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu mpirun -np 1 --bind-to none env OMP_NUM_THREADS=8 TP_OMP_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1 DENDRO_TIME_REMESH=1 $BIN local.par.toml 1 > run.log 2>&1
    echo "$name cap=$cap tol=$tol gpu=$gpu wall=$(( $(date +%s)-s ))s $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1)" >> $OUT/summary_T${TEND}.txt ) &
done
wait
echo RESSTUDY2_DONE >> $OUT/summary_T${TEND}.txt
