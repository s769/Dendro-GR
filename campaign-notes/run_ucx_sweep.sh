#!/usr/bin/env bash
export SKIP_BITID=1
S=$HOME/Dendro-GR/campaign-notes/run_ab_env.sh
bash $S ucx1 "UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy" "UCX_TLS=cuda_ipc,cuda_copy,sm,self" 1 4
bash $S ucx2 "UCX_RNDV_THRESH=8192" "UCX_RNDV_THRESH=inf" 1 4
bash $S ucx3 "UCX_RNDV_SCHEME=auto" "UCX_RNDV_SCHEME=put_zcopy" 1 4
echo UCX_SWEEP_DONE
