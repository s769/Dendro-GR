#!/usr/bin/env bash
# Baselines (handoff §5.1) + Tier 1.5 dual-res v2. Run on a GPU node.
set -uo pipefail
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null
unset CUDA_VISIBLE_DEVICES; export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
N="$HOME/Dendro-GR/campaign-notes"; L="$N/tier0/verify_baselines_$(hostname).log"
B="$DENDRO_BUILD/BSSN_GR"; cd "$B"
echo "host $(hostname) $(date -Is)" | tee "$L"; nvidia-smi -L | head -1 | tee -a "$L"
run() { echo "== $1 ==" | tee -a "$L"; shift; env "$@" ./rhsgpu_test 2048 20 0 2>&1 | grep -E 'gpu_time|time|l_inf|max_rel|mismatch' | head -14 | tee -a "$L"; }
run stock          DENDRO_NONE=1
run PT_ASYNC       DENDRO_TIME_PT_ASYNC=1
run PT_ASYNC_WHILEHOT DENDRO_TIME_PT_ASYNC_WHILEHOT=1
run PT_F32SU       DENDRO_TIME_PT_F32SU=1
echo "== dual-res v2 ==" | tee -a "$L"
bash "$N/run_tier15_dualres_v2.sh" 2>&1 | tee -a "$L" | tail -15
echo VERIFY_DONE | tee -a "$L"
