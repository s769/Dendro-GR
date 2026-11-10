#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
export RMODE=4 GWF=10
for cfg in "R1|DENDRO_GPU_BH_WAMR=1 DENDRO_REMESH_DOUBLE_UPLOAD=1" "R2|DENDRO_GPU_BH_WAMR=1" "R3|DENDRO_GPU_BH_WAMR=1 DENDRO_GPU_WAMR=1"; do
  tag=${cfg%%|*}; envs=${cfg#*|}
  echo "=== $tag: $envs"
  bash ./run_remesh_time.sh $tag "$envs" > /dev/null 2>&1
  f=$HOME/Dendro-GR-runs/rm_$tag/run.log
  sed 's/\x1b\[[0-9;]*m//g' $f | grep -E "ETS time" | tail -1
  sed 's/\x1b\[[0-9;]*m//g' $f | grep -E "remesh-time" | tail -1
  sed 's/\x1b\[[0-9;]*m//g' $f | grep -E "remesh-phases" | head -3
  sed 's/\x1b\[[0-9;]*m//g' $f | grep -E "grid-transfer-phases" | head -2
  echo "remeshes: $(sed 's/\x1b\[[0-9;]*m//g' $f | grep -c 'Remesh is triggered')"
done
python3 - <<'PY'
import re,numpy as np,os
def load(d,lm):
    z=[]
    for l in open(os.path.expanduser(f'~/Dendro-GR-runs/{d}/bssn_profile_GW_{lm}.dat')).read().splitlines():
        p=l.split()
        try: float(p[0]); float(p[1])
        except: continue
        if len(p)>2: z.append([complex(*map(float,re.findall(r'[-+0-9.eE]+',c))) for c in p[2:]])
    return np.array(z)
for m in ['rm_R2','rm_R3']:
    a=load('rm_R1','l2_m2'); b=load(m,'l2_m2'); n=min(len(a),len(b))
    print(m,'vs R1 l2_m2: rows',len(a),len(b),'max abs diff %.2e (|a|max %.2e)'%(abs(a[:n]-b[:n]).max(),abs(a).max()))
PY
echo RMAB_DONE
