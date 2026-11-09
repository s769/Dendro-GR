#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
for t in A2 A3; do d=$HOME/Dendro-GR-runs/ab_bdystream_q1_$t; mkdir -p $d/{aeh,cp,dat,vtu}; cd $d
 ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin rit_q1_tpid_sol.bin
 mpirun -np 1 env DENDRO_RHS_PT_ASYNC=1 DENDRO_BDY_STREAM=0 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA $HOME/Dendro-GR-runs/q1.smoke_cons.par.toml 1 > run.log 2>&1; done; echo REP_DONE
