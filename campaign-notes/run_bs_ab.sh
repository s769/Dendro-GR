#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes; export RMODE=4 GWF=10
bash ./run_remesh_time.sh V 'DENDRO_GPU_BH_WAMR=1 DENDRO_GPU_WAMR=1 DENDRO_BLKSETUP_VERIFY=1' 8 >/dev/null 2>&1
bash ./run_remesh_time.sh B 'DENDRO_GPU_BH_WAMR=1 DENDRO_GPU_WAMR=1' 8 >/dev/null 2>&1
echo "verify mismatch lines: $(sed 's/\x1b\[[0-9;]*m//g' $HOME/Dendro-GR-runs/rm_V/run.log | strings | grep -c 'blksetup-verify')"
for t in V B; do f=$HOME/Dendro-GR-runs/rm_$t/run.log
 echo "== $t: $(sed 's/\x1b\[[0-9;]*m//g' $f | strings | grep 'ETS time' | tail -1)  remeshes=$(sed 's/\x1b\[[0-9;]*m//g' $f | strings | grep -c 'Remesh is triggered')"
 sed 's/\x1b\[[0-9;]*m//g' $f | strings | grep 'remesh-inner' | tail -1 | cut -c1-120
 sed 's/\x1b\[[0-9;]*m//g' $f | strings | grep 'remesh-phases' | tail -1 | cut -c1-140; done
python3 - <<'PY'
import re,numpy as np,os
def load(d,lm):
    z=[]
    for l in open(os.path.expanduser(f'~/Dendro-GR-runs/{d}/bssn_profile_GW_{lm}.dat')).read().splitlines():
        p=l.split()
        if not p or not re.match(r'^[0-9]+$',p[0]): continue
        if len(p)>2: z.append([complex(*map(float,re.findall(r'[-+0-9.eE]+',c))) for c in p[2:]])
    return np.array(z)
for lm in ['l2_m2','l3_m2']:
    a=load('rm_S1',lm); b=load('rm_B',lm); n=min(len(a),len(b))
    print(lm,'S1(serial, tree search) vs B rows',len(a),len(b),'max abs diff %.2e (|a|max %.2e)'%(abs(a[:n]-b[:n]).max(),abs(a).max()))
PY
echo BS_DONE
