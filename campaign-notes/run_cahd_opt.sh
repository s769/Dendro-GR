#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null
rm -f $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA   # a build killed mid-link can leave a truncated binary that make treats as up to date
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | tail -1 ) > tier0/cahdopt_build.log 2>&1
# equivalence check vs the CPU (stage-2 RHS + states), same harness as before
STEP=0 RK=2 TEND=0.1 bash ./run_statecmp_cahd_step.sh
# production-depth timing with the optimized CAHD pass (new defaults, CAHD on via q8 par)
E='DENDRO_GW_SHELL_ONLY=1 DENDRO_TIME_REMESH=1'
OMPT=36 bash ./run_prod_q8.sh CAHDFUS 0.3 "$E" 0 > tier0/prod_CAHDFUS.out 2>&1
OMPT=36 bash ./run_prod_q8.sh CAHDREF 0.3 "$E DENDRO_GPU_CAHD_FUSED=0" 0 > tier0/prod_CAHDREF.out 2>&1
OMPT=36 bash ./run_prod_q8.sh CAHDOFF 0.3 "$E DENDRO_GPU_CAHD=0" 0 > tier0/prod_CAHDOFF.out 2>&1
echo D > tier0/cahdopt.done
