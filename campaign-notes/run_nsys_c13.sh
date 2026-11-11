#!/usr/bin/env bash
# nsys profile of the production q1 run at CAP (default 13) for TEND (default 14) on one GPU; prints kernel + CUDA API summaries.
# usage: [CAP=13 TEND=14 CFL=0.25 RMF=50 GPU=0 LABEL=base] run_nsys_c13.sh
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
CAP=${CAP:-13}; TEND=${TEND:-14}; CFL=${CFL:-0.25}; RMF=${RMF:-50}; GPU=${GPU:-0}; LABEL=${LABEL:-base}
NSYS=/home1/apps/nvidia/Linux_aarch64/26.9/compilers/bin/nsys
d=$HOME/Dendro-GR-runs/nsys13_$LABEL; rm -rf $d; mkdir -p $d/{aeh,bah,cp,dat,vtu}; cd $d
cp $HOME/Dendro-GR/BSSN_GR/pars/q1.par.toml local.par.toml; ln -sf $HOME/Dendro-GR-runs/resstudy2/tp/tp_q001_tpid_sol.bin .
python3 - <<PY
import re
from pathlib import Path
p=Path("local.par.toml"); t=p.read_text()
def sub(k,v):
    global t
    t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M)
    if n==0:
        m=re.search(r"^\[",t,flags=re.M); ins=k+" = "+v+"\n\n"; t=(t[:m.start()]+ins+t[m.start():]) if m else t+"\n"+ins
for k,v in [("BSSN_RESTORE_SOLVER","0"),("BSSN_MAXDEPTH","$CAP"),("BSSN_BH1_MAX_LEV","$CAP"),("BSSN_BH2_MAX_LEV","$CAP"),("BSSN_RK_TIME_END","$TEND.0"),("BSSN_CFL_FACTOR","$CFL"),
            ("BSSN_REMESH_TEST_FREQ","$RMF"),("BSSN_GW_EXTRACT_FREQ","$((8*(1<<(CAP-11))))"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0")]: sub(k,v)
p.write_text(t)
PY
CUDA_VISIBLE_DEVICES=$GPU $NSYS profile -o $d/rep --force-overwrite=true --trace=cuda,nvtx --stats=false mpirun -np 1 --bind-to none env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
$NSYS stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum --format csv rep.nsys-rep > stats.csv 2>/dev/null
$NSYS stats --report cuda_gpu_kern_sum rep.nsys-rep 2>/dev/null | head -30 > kern_top.txt
echo NSYS_DONE >> $d/status.txt
