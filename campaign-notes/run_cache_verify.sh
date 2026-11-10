#!/usr/bin/env bash
source /opt/apps/lmod/lmod/init/bash 2>/dev/null || true
source "$HOME/Dendro-GR/campaign-notes/env_build_horizon.sh" >/dev/null; unset CUDA_VISIBLE_DEVICES
BIN=$DENDRO_BUILD/BSSN_GR/bssnSolverCUDA
d=$HOME/Dendro-GR-runs/q8x_verify; mkdir -p $d; find $d -mindepth 1 -delete 2>/dev/null; mkdir -p $d/{aeh,cp,dat,vtu}
cp $HOME/Dendro-GR-runs/q8x_base/local.par.toml $d/local.par.toml; ln -sfn $HOME/Dendro-GR-runs/q8_hist_4gpu/mesh_q8_tpid_sol.bin $d/mesh_q8_tpid_sol.bin; cd $d
mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=2 env OMP_NUM_THREADS=2 DENDRO_RHS_PT_ASYNC=1 DENDRO_GW_SHELL_ONLY=0 DENDRO_GW_CACHE_SAMPLER=1 DENDRO_GW_CACHE_VERIFY=1 $BIN local.par.toml 1 > run.log 2>&1
grep "gw-cache-verify" run.log | awk '{print $0}' > verify.txt; wc -l verify.txt
grep "same_idx 0" verify.txt | head -3; awk '{ if ($NF+0 > m) m=$NF+0 } END {print "max |dR| over all:", m}' verify.txt
grep -c "same_idx 0" verify.txt
sed 's/\x1b\[[0-9;]*m//g' run.log | grep -n "Remesh is triggered\|gw-cache-verify" | awk -F: '{print $1": "substr($0,index($0,$2),70)}' | sed -n 1,3p
