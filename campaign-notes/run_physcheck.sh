#!/usr/bin/env bash
# GPU (adaptive BH_WAMR) vs CPU vs GPU(no remesh) physics check. usage: run_physcheck.sh cpu|gpu|gpu0 TEND
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
MODE=$1; TEND=${2:-20.0}
d=$HOME/Dendro-GR-runs/phys_$MODE; mkdir -p $d; find $d -mindepth 1 -delete 2>/dev/null; mkdir -p $d/{aeh,cp,dat,vtu}
cp $HOME/Dendro-GR-runs/q1.smoke_cons.par.toml $d/local.par.toml
python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
def sub(k,v):
    global t; t,n=re.subn(r"^"+k+r" = .*$",k+" = "+v,t,flags=re.M)
    if n==0: t=t+"\n"+k+" = "+v+"\n"
for k,v in [("BSSN_MAXDEPTH","13"),("BSSN_BH1_MAX_LEV","13"),("BSSN_BH2_MAX_LEV","13"),("BSSN_RK_TIME_END","$TEND"),
            ("BSSN_REMESH_TEST_FREQ","0" if "$MODE" in ("cpu0",) or "$MODE".startswith("gf") else "10"),("BSSN_GW_EXTRACT_FREQ","10"),("BSSN_REFINEMENT_MODE","4"),
            ("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0"),("BSSN_TIME_STEP_OUTPUT_FREQ","10"),
            ("BSSN_GW_RADAII","[10.0, 12.0, 15.0, 18.0, 20.0, 25.0]")]: sub(k,v)
p.write_text(t)
PY
ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
S=$(date +%s)
case $MODE in
  cpu0) mpirun -np 36 --bind-to core env OMP_NUM_THREADS=1 $DENDRO_BUILD/BSSN_GR/bssnSolver local.par.toml 1 > run.log 2>&1 ;;
  cpu)  mpirun -np 36 --bind-to core env OMP_NUM_THREADS=1 $DENDRO_BUILD/BSSN_GR/bssnSolver local.par.toml 1 > run.log 2>&1 ;;
  gpu)  mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=8 env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1 ;;
  gpu0) mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=8 env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GPU_BH_WAMR=0 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1 ;;
  gf*)  mpirun -np ${NP:-4} --bind-to none -x OMP_NUM_THREADS=8 env OMP_NUM_THREADS=8 DENDRO_GPU_BH_WAMR=0 $EXTRA $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1 ;;
esac
echo "$MODE wall=$(( $(date +%s)-S ))s $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1) remeshes=$(sed 's/\x1b\[[0-9;]*m//g' run.log | grep -c 'Remesh is triggered')"
