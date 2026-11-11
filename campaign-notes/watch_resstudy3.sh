#!/usr/bin/env bash
# whichever of the pending allocations (89094 5h, 89140 2h) starts first runs production-like batch 3; the other is cancelled
D=$HOME/Dendro-GR/campaign-notes
while true; do
  for j in 89094 89140; do
    N=$(squeue -j $j -h -t R -o %N 2>/dev/null | head -1)
    if [ -n "$N" ]; then
      other=$([ $j = 89094 ] && echo 89140 || echo 89094); scancel $other 2>/dev/null
      sleep 15
      ssh -o StrictHostKeyChecking=no $N "nohup env TEND=110 bash $D/run_resstudy2.sh 'p13:13:0.00001:0 g3c12:12:0.00001:1:BSSN_GW_REFINE_WTOL=0.001 g5c12:12:0.00001:2:BSSN_GW_REFINE_WTOL=0.00001' > $D/tier0/resstudy3a.log 2>&1 &"
      ssh -o StrictHostKeyChecking=no $N "nohup env TEND=80 bash $D/run_resstudy2.sh 'p14:14:0.00001:3' > $D/tier0/resstudy3b.log 2>&1 &"
      echo "$(date -Is) batch 3 launched on $N (job $j; cancelled $other)" >> $D/tier0/watch_resstudy3.log
      exit 0
    fi
  done
  sleep 20
done
