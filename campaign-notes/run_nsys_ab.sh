#!/usr/bin/env bash
# usage: run_nsys_ab.sh LABEL "ENVS" [np=4]  -> nsys of q8 meshprobe TIME_END=0.5
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=1
L="$1"; E="$2"; NP="${3:-4}"; d=$HOME/Dendro-GR-runs/nsys_$L; mkdir -p $d/{aeh,cp,dat,vtu}; cd $d
cp $HOME/Dendro-GR-runs/q8.meshprobe.par.toml local.par.toml
python3 - <<PY
from pathlib import Path; import re
p=Path("local.par.toml"); t=p.read_text()
for k,v in [("BSSN_RK_TIME_END","0.5"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0")]: t=re.sub(k+r" = [0-9.eE+-]+",k+" = "+v,t)
p.write_text(t)
PY
ln -sfn $HOME/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin mesh_q8_tpid_sol.bin
nsys profile -o $d/rep --force-overwrite=true --trace=cuda,nvtx --stats=false \
  mpirun -np $NP env -u CUDA_VISIBLE_DEVICES DENDRO_RHS_PT_ASYNC=1 $E $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
echo "NSYS_DONE $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time (max)' | tail -1)"
