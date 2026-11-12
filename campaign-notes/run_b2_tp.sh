#!/usr/bin/env bash
# usage: run_b2_tp.sh <param> [delta]   (builds tpTangent, runs in a copy-free rundir that holds the solved tp file)
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source $HOME/Dendro-GR/campaign-notes/env_build_horizon.sh >/dev/null
cd $DENDRO_BUILD && cmake . >/dev/null 2>&1; make -j16 tpTangent 2>&1 | grep -E "error|Built target tpTangent" | head -20
cd $HOME/Dendro-GR-runs/resstudy2/p14N_T230 && $DENDRO_BUILD/BSSN_GR/tpTangent local.par.toml "$@"
