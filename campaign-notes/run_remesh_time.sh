#!/usr/bin/env bash
# remesh-test timing on q8 meshprobe (depth 15, 4 ranks): usage run_remesh_time.sh TAG "ENVS" [OMP]
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
TAG=$1; ENVS=$2; OMPT=${3:-1}; BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
d=$HOME/Dendro-GR-runs/rm_$TAG; mkdir -p $d; find $d -mindepth 1 -delete 2>/dev/null; mkdir -p $d/{aeh,cp,dat,vtu}
cp $HOME/Dendro-GR-runs/q8.meshprobe.par.toml $d/local.par.toml
python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
for k,v in [("BSSN_RK_TIME_END","0.5"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0"),("BSSN_GW_EXTRACT_FREQ","0"),("BSSN_REMESH_TEST_FREQ","10"),("BSSN_REFINEMENT_MODE","${RMODE:-3}")]:
    t=re.sub(k+r" = [0-9.eE+-]+",k+" = "+v,t)
p.write_text(t)
PY
ln -sfn $HOME/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin $d/mesh_q8_tpid_sol.bin; cd $d
mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=$OMPT env OMP_NUM_THREADS=$OMPT DENDRO_RHS_PT_ASYNC=1 DENDRO_TIME_REMESH=1 $ENVS $BIN local.par.toml 1 > run.log 2>&1
sed 's/\x1b\[[0-9;]*m//g' run.log | grep -E "remesh-time|ETS time|Remesh is triggered" | tail -4 | cut -c1-150
sed 's/\x1b\[[0-9;]*m//g' run.log | grep -c "Remesh is triggered"
