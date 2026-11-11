#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=1
d=$HOME/Dendro-GR-runs/unzipcmp; mkdir -p $d/{aeh,cp,dat,vtu}; cp $HOME/Dendro-GR-runs/exab_full/local.par.toml $d/local.par.toml
python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text(); t=re.sub(r"BSSN_RK_TIME_END = [0-9.eE+-]+","BSSN_RK_TIME_END = 0.3",t); p.write_text(t)
PY
ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
CUDA_VISIBLE_DEVICES=0 mpirun -np 1 --bind-to none env DENDRO_RHS_PT_ASYNC=1 DENDRO_UNZIP_CMP=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
grep "unzip-cmp" run.log
