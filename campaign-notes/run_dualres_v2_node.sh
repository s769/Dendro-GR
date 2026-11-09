#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
unset CUDA_VISIBLE_DEVICES
bash $HOME/Dendro-GR/campaign-notes/run_tier15_dualres_scan.sh > $HOME/Dendro-GR/campaign-notes/tier0/dualres_scan_$(hostname).log 2>&1
