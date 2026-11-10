#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null; unset CUDA_VISIBLE_DEVICES
S=$HOME/Dendro-GR-runs/statecmp_cahd_rk2/gpu
for V in 0 1; do
  d=$HOME/Dendro-GR-runs/cahd_s2_$V; rm -rf $d; mkdir -p $d/{aeh,cp,dat,vtu,dump}; cp $S/local.par.toml $d/
  ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
  sed -i -E 's/^BSSN_RK_TIME_END = .*/BSSN_RK_TIME_END = 0.04/' local.par.toml
  mpirun -np 8 --bind-to none -x OMP_NUM_THREADS=1 -x DENDRO_DUMP_RHS -x DENDRO_GPU_CAHD env DENDRO_DUMP_RHS=$d/dump:0:2 DENDRO_GPU_CAHD=$V DENDRO_RHS_PT_ASYNC=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
done
echo D > $HOME/Dendro-GR/campaign-notes/tier0/cahd_s2.done
