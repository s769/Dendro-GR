#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source $HOME/Dendro-GR/campaign-notes/env_build_horizon.sh >/dev/null
cd $DENDRO_BUILD && cmake . >/dev/null 2>&1; make -j16 bssnTaylorRef 2>&1 | grep -E "error|Built target bssnTaylorRef"
cd $DENDRO_BUILD/BSSN_GR && OMP_NUM_THREADS=1 TAYLOR_TIMING=1 ./bssnTaylorRef 2>&1 | grep -E "^\(A\)|^\(T\)|^    "
