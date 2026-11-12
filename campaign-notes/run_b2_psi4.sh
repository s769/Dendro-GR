#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source $HOME/Dendro-GR/campaign-notes/env_build_horizon.sh >/dev/null
cd $DENDRO_BUILD && cmake . >/dev/null 2>&1; make -j16 bssnTaylorPsi4 2>&1 | grep -E "error|Error|Built target bssnTaylorPsi4" | head -20
./BSSN_GR/bssnTaylorPsi4
