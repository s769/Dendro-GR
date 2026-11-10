#!/usr/bin/env bash
# build init-refmode fix, then frozen-mesh GPU vs CPU physics check (CPU ref = phys_cpu0)
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | tail -3 ) > tier0/fix_build.log 2>&1
NP=1 EXTRA="DENDRO_RHS_PT_ASYNC=1" bash ./run_physcheck.sh gf_fix 6.4 > tier0/phys_gf_fix.out 2>&1
echo DONE > tier0/fixcheck.done
