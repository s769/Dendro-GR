#!/usr/bin/env bash
# Resume a checkpointed production run on the current node: usage: resume_run.sh RUNDIR GPU   (RUNDIR contains local.par.toml, cp/)
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
d=$1; g=${2:-0}; cd $d || exit 1
sed -i 's/^BSSN_RESTORE_SOLVER = .*/BSSN_RESTORE_SOLVER = 1/' local.par.toml
n=$(ls run*.log 2>/dev/null | wc -l); mv run.log run.log.$n 2>/dev/null
CUDA_VISIBLE_DEVICES=$g mpirun -np 1 --bind-to none env OMP_NUM_THREADS=8 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=1 DENDRO_BH_TRACK_FREQ=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
echo "resumed run finished: $(sed 's/\x1b\[[0-9;]*m//g' run.log | grep 'ETS time' | tail -1)" >> $d/resume_status.txt
