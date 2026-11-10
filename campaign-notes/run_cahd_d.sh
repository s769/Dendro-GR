#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null
rm -f $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | grep -E "error|Built target bssnSolverCUDA" | tail -5 ) > tier0/cahdd_build.log 2>&1
CU=/home1/apps/nvidia/Linux_aarch64/26.9/cuda/13.3/bin/cuobjdump
$CU -res-usage $DENDRO_BUILD/BSSN_GR/bssnSolverCUDA 2>/dev/null | grep -A1 -E "cahd_fused_c" | grep -E "Function|REG" | paste - - | sed -E 's/Function ([^ ]{0,70}).*REG:([0-9]+) STACK:([0-9]+) SHARED:([0-9]+) LOCAL:([0-9]+).*/\1 REG=\2 STACK=\3/' > tier0/cahdd_regs.txt
rm -f tier0/statecmp_cahd_s0_rk2.done
STEP=0 RK=2 TEND=0.1 bash ./run_statecmp_cahd_step.sh
E='DENDRO_GW_SHELL_ONLY=1 DENDRO_TIME_REMESH=1'
OMPT=36 bash ./run_prod_q8.sh CAHDD2 0.3 "$E" 0 > tier0/prod_CAHDD2.out 2>&1
OMPT=36 bash ./run_prod_q8.sh CAHDD3 0.3 "$E DENDRO_GPU_CAHD_MODE=3" 0 > tier0/prod_CAHDD3.out 2>&1
OMPT=36 bash ./run_prod_q8.sh CAHDD0 0.3 "$E DENDRO_GPU_CAHD=0" 0 > tier0/prod_CAHDD0.out 2>&1
echo D > tier0/cahdd.done
