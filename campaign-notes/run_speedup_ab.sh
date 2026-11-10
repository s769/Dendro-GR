#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
run() { # tag "envs"
  d=$HOME/Dendro-GR-runs/su_$1; mkdir -p $d; find $d -mindepth 1 -delete 2>/dev/null; mkdir -p $d/{aeh,cp,dat,vtu}
  cp $HOME/Dendro-GR-runs/q8.meshprobe.par.toml $d/local.par.toml
  python3 - <<PY
from pathlib import Path; import re
p=Path("$d/local.par.toml"); t=p.read_text()
for k,v in [("BSSN_RK_TIME_END","0.5"),("BSSN_IO_OUTPUT_FREQ","0"),("BSSN_CHECKPT_FREQ","0")]: t=re.sub(k+r" = [0-9.eE+-]+",k+" = "+v,t)
p.write_text(t)
PY
  ln -sfn $HOME/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin $d/mesh_q8_tpid_sol.bin; cd $d
  mpirun -np 4 --bind-to none env $2 $BIN local.par.toml 1 > run.log 2>&1
  echo "$1: ETS=$(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time (max)' | tail -1 | awk '{print $NF}')"
}
for i in 1 2; do
  run S0_$i "DENDRO_BDY_STREAM=0"                                   # stock kernel baseline (no PT_ASYNC)
  run S1_$i "DENDRO_RHS_PT_ASYNC=1 DENDRO_BDY_STREAM=0"             # + PT_ASYNC (lb(343,1) compiled in)
  run S2_$i "DENDRO_RHS_PT_ASYNC=1"                                 # + boundary overlap (current default)
  run S3_$i "DENDRO_RHS_PT_ASYNC=1 DENDRO_RHS_PT_ASYNC_WHILEHOT=1"  # + opt-in while-hot
done
echo SPEEDUP_DONE
