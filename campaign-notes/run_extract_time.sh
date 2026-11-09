#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=1
d=$HOME/Dendro-GR-runs/extime; mkdir -p $d/{aeh,cp,dat,vtu}; cp $HOME/Dendro-GR-runs/q1.smoke_cons.par.toml $d/local.par.toml
python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+","BSSN_RK_TIME_END = 4.0",t)
t=re.sub(r"BSSN_REMESH_TEST_FREQ = [0-9]+","BSSN_REMESH_TEST_FREQ = 20",t)
t=re.sub(r"BSSN_GW_EXTRACT_FREQ = [0-9]+","BSSN_GW_EXTRACT_FREQ = 4",t)
t=re.sub(r"BSSN_GW_RADAII = \[[^\]]*\]","BSSN_GW_RADAII = [10.0,12.0,15.0,18.0,20.0,25.0]",t)
p.write_text(t)
PY
ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
CUDA_VISIBLE_DEVICES=0 mpirun -np 1 env DENDRO_RHS_PT_ASYNC=1 DENDRO_TIME_EXTRACT=1 $EXTRA_ENV $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
grep -E "extract-time" run.log | tail -4; sed 's/\x1b\[[0-9;]*m//g' run.log | grep "ETS time (max)"
