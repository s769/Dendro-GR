#!/usr/bin/env bash
# effect of CAHD (C=0.06, as in production par files): CPU(with CAHD) vs CPU(no CAHD)=phys_cpu0 vs GPU(ignores CAHD)
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | tail -2 ) > tier0/cahd_build.log 2>&1
PAR_OVR="BSSN_CAHD_C=0.06" NP=1 bash ./run_physcheck.sh fzgpu_cahd 6.4 > tier0/phys_fzgpu_cahd.out 2>&1 &
PAR_OVR="BSSN_CAHD_C=0.06" bash ./run_physcheck.sh fzcpu_cahd 6.4 > tier0/phys_fzcpu_cahd.out 2>&1
wait
echo D > tier0/cahd_batch.done
rm -f tier0/timing_batch.done; bash ./run_timing_batch.sh
