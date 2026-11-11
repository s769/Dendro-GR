#!/usr/bin/env bash
D=$HOME/Dendro-GR/campaign-notes
launch() { # alloc  tend  specs  logname
  until [ -n "$(squeue -j $1 -h -t R -o %N 2>/dev/null)" ]; do sleep 20; done
  N=$(squeue -j $1 -h -t R -o %N | head -1); sleep 15
  ssh -o StrictHostKeyChecking=no $N "nohup env TEND=$2 bash $D/run_resstudy2.sh '$3' > $D/tier0/$4.log 2>&1 &"
  echo "$(date -Is) $4 launched on $N (alloc $1)" >> $D/tier0/watch_rs3.log
}
launch 89173 110 'p13:13:0.00001:0 g3c12:12:0.00001:1:BSSN_GW_REFINE_WTOL=0.001 g5c12:12:0.00001:2:BSSN_GW_REFINE_WTOL=0.00001 a4c12:12:0.00001:3:BSSN_DENDRO_AMR_FAC=0.4' rs3a &
launch 89174 80 'p14:14:0.00001:0' rs3b &
wait
