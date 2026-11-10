#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null
rm -f $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | grep -E "error|Built target" | tail -5 ) > tier0/cahdc_build.log 2>&1
cp $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA $HOME/Dendro-GR-runs/bssnSolverCUDA.cahdc 2>/dev/null
# 1) equivalence with the CPU (stage-2 RHS and states), default mode = 2 (contracted)
STEP=0 RK=2 TEND=0.1 bash ./run_statecmp_cahd_step.sh
# 2) production-depth timing: contracted (mode 2), generated fused (mode 1), CAHD off
E='DENDRO_GW_SHELL_ONLY=1 DENDRO_TIME_REMESH=1'
OMPT=36 bash ./run_prod_q8.sh CAHDC2 0.3 "$E" 0 > tier0/prod_CAHDC2.out 2>&1
OMPT=36 bash ./run_prod_q8.sh CAHDC1 0.3 "$E DENDRO_GPU_CAHD_MODE=1" 0 > tier0/prod_CAHDC1.out 2>&1
OMPT=36 bash ./run_prod_q8.sh CAHDC0 0.3 "$E DENDRO_GPU_CAHD=0" 0 > tier0/prod_CAHDC0.out 2>&1
echo D > tier0/cahdc.done
