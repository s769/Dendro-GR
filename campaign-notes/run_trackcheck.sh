#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source ./env_build_horizon.sh >/dev/null
( cd $DENDRO_BUILD && make -j16 bssnSolverCUDA 2>&1 | tail -2 ) > tier0/track_build.log 2>&1
NP=1 EXTRA="DENDRO_RHS_PT_ASYNC=1 DENDRO_BH_TRACK_FREQ=1" bash ./run_physcheck.sh gf_track1 6.4 > tier0/phys_gf_track1.out 2>&1
NP=1 EXTRA="DENDRO_RHS_PT_ASYNC=1 DENDRO_BH_TRACK_FREQ=5" bash ./run_physcheck.sh gf_track5 6.4 > tier0/phys_gf_track5.out 2>&1
echo D > tier0/track.done
