#!/usr/bin/env bash
# Extra FD members for second derivatives (p=2: bare masses of BH1, BH2) on the fdtp_c$CAP fiducial mesh record.
# usage: CAP=13 GPUS="1 2 3" MEMBERS="b2p:0:+0.001 b2m:0:-0.001 pp:+0.001:+0.001 ..." run_fd_hess.sh   (tag:dm1:dm2)
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
CAP=${CAP:-13}; TEND=${TEND:-110}; GPUS=(${GPUS:-0 1 2 3}); M0=0.5
OUT=$HOME/Dendro-GR-runs/fdtp_c$CAP; REC=$OUT/mesh_rec
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA; TPID=$DENDRO_BUILD/BSSN_GR/tpid
TPB=$HOME/Dendro-GR-runs/resstudy2/tp/tp_q001_tpid_sol.bin
gwf=$(( 8 * (1 << (CAP - 11)) ))
setup() { # tag m1 m2
  local d=$OUT/$1; rm -rf $d; mkdir -p $d/{aeh,bah,cp,dat,vtu}; cp $HOME/Dendro-GR/BSSN_GR/pars/q1.par.toml $d/local.par.toml
  python3 - <<PY
import re
from pathlib import Path
p=Path("$d/local.par.toml"); t=p.read_text()
t=re.sub(r"(BSSN_BH1 = \{ MASS = )[0-9.eE+-]+", r"\g<1>$2", t, count=1)
t=re.sub(r"(BSSN_BH2 = \{ MASS = )[0-9.eE+-]+", r"\g<1>$3", t, count=1)
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
for spec in $MEMBERS; do                       # TP solves first (CPU), sequential
  IFS=: read tag d1 d2 <<< "$spec"
  m1=$(python3 -c "print($M0 $d1)"); m2=$(python3 -c "print($M0 $d2)")
  setup $tag $m1 $m2; d=$OUT/$tag
  ( cd $d; env OMP_NUM_THREADS=36 TP_OMP_THREADS=36 DENDRO_TPID_WARM_START=$TPB $TPID local.par.toml 36 > tpid.log 2>&1; echo "$tag TP: $(grep 'TP solver time' tpid.log | tail -1) m1=$m1 m2=$m2" >> $OUT/summary_hess.txt )
  [ -f $d/tp_q001_tpid_sol.bin ] || echo "TP SOLVE FAILED for $tag" >> $OUT/summary_hess.txt
done
i=0
for spec in $MEMBERS; do
  IFS=: read tag d1 d2 <<< "$spec"; g=${GPUS[$((i % ${#GPUS[@]}))]}
  ( cd $OUT/$tag; s=$(date +%s); CUDA_VISIBLE_DEVICES=$g mpirun -np 1 --bind-to none env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1 DENDRO_MESH_REPLAY=$REC $BIN local.par.toml 1 > run.log 2>&1
    echo "$tag wall=$(( $(date +%s)-s ))s $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1)" >> $OUT/summary_hess.txt ) &
  i=$((i+1)); if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done; wait; echo HESS_DONE >> $OUT/summary_hess.txt
