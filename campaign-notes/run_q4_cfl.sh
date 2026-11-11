#!/usr/bin/env bash
# q=4 CFL/remesh-interval stability+accuracy check at cap CAP, T=TEND. variants "tag:cfl:rmf:gpu"
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
CAP=${CAP:-14}; TEND=${TEND:-50}; OUT=$HOME/Dendro-GR-runs/q4cfl; mkdir -p $OUT/tp
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
if [ ! -f $OUT/tp/tp_q004_tpid_sol.bin ]; then
  cp $HOME/Dendro-GR/BSSN_GR/pars/q4.par.toml $OUT/tp/q4.par.toml; ( cd $OUT/tp; mkdir -p aeh bah cp dat vtu; env OMP_NUM_THREADS=36 TP_OMP_THREADS=36 $DENDRO_BUILD/BSSN_GR/tpid q4.par.toml 36 > tpid.log 2>&1 )
fi
[ -f $OUT/tp/tp_q004_tpid_sol.bin ] || { echo "TP FAILED" >> $OUT/summary.txt; exit 1; }
for spec in $VARIANTS; do
  IFS=: read tag c rmf gpu <<< "$spec"; d=$OUT/${tag}_c${CAP}_T${TEND}; rm -rf $d; mkdir -p $d/{aeh,bah,cp,dat,vtu}
  cp $HOME/Dendro-GR/BSSN_GR/pars/q4.par.toml $d/local.par.toml; ln -sfn $OUT/tp/tp_q004_tpid_sol.bin $d/tp_q004_tpid_sol.bin
  gwf=$(python3 -c "print(max(1,round(8*2**($CAP-11)*0.25/$c)))")
  python3 - <<PY
import re
from pathlib import Path
p=Path("$d/local.par.toml"); t=p.read_text()
def sub(k,v):
    global t
    t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M)
    if n==0:
        m=re.search(r"^\[",t,flags=re.M); ins=k+" = "+v+"\n\n"; t=(t[:m.start()]+ins+t[m.start():]) if m else t+"\n"+ins
for k,v in [("BSSN_RESTORE_SOLVER","0"),("BSSN_MAXDEPTH","$CAP"),("BSSN_BH1_MAX_LEV","$CAP"),("BSSN_BH2_MAX_LEV","$CAP"),("BSSN_RK_TIME_END","$TEND.0"),
            ("BSSN_CFL_FACTOR","$c"),("BSSN_REMESH_TEST_FREQ","$rmf"),("BSSN_GW_EXTRACT_FREQ","$gwf"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0")]: sub(k,v)
p.write_text(t)
PY
  ( cd $d; s=$(date +%s); CUDA_VISIBLE_DEVICES=$gpu mpirun -np 1 --bind-to none env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1 $BIN local.par.toml 1 > run.log 2>&1
    echo "$tag cfl=$c rmf=$rmf wall=$(( $(date +%s)-s ))s $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1) | $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep -o 'current time : *[0-9.]*' | tail -1)" >> $OUT/summary.txt ) &
done; wait; echo Q4_DONE >> $OUT/summary.txt
