#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null; unset CUDA_VISIBLE_DEVICES
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | tail -1 ) > tier0/wsdump_build.log 2>&1
S=$HOME/Dendro-GR-runs/statecmp_cahd_s0_rk2/gpu
d=$HOME/Dendro-GR-runs/cahd_wsdump; rm -rf $d; mkdir -p $d/{aeh,cp,dat,vtu,dump}; cp $S/local.par.toml $d/
ln -sfn $HOME/Dendro-GR-runs/tpid_work/rit_q1_tpid_sol.bin $d/rit_q1_tpid_sol.bin; cd $d
sed -i -E "s/^BSSN_RK_TIME_END = .*/BSSN_RK_TIME_END = 0.04/" local.par.toml
# rank 0 owns global ids 0..939 of its local list; block 598 in rank-0 numbering
mpirun -np 8 --bind-to none -x OMP_NUM_THREADS=1 -x DENDRO_CAHD_DUMPBLK env DENDRO_CAHD_DUMPBLK=598:2 DENDRO_RHS_PT_ASYNC=1 $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA local.par.toml 1 > run.log 2>&1
echo D > $HOME/Dendro-GR/campaign-notes/tier0/wsdump.done
