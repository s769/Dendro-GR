#!/usr/bin/env bash
D=$HOME/Dendro-GR/campaign-notes
until [ -n "$(squeue -j 

Horizon


this
submit
jobname...OK
ssh
to
allocation
quota
88861 -h -t R -o %N)" ]; do sleep 10; done
N=$(squeue -j 

Horizon


this
submit
jobname...OK
ssh
to
allocation
quota
88861 -h -t R -o %N); sleep 15
echo "$(date -Is) launching fine replays v2 on $N" >> $D/tier0/fine_replays.log
R='[10.0, 12.0, 15.0, 18.0, 20.0, 25.0]'
ssh -o StrictHostKeyChecking=no $N "nohup env SKIP_FID=1 OMPT=36 XENV='DENDRO_GW_SHELL_ONLY=1' RES=fine TEND=26 RFREQ=20 OUTTAG=_cahd PAR_OVR='BSSN_CAHD_C=0.06' RADII='$R' bash $D/run_tier15_waveform_fd.sh > $D/tier0/wave_fine_cahd_rep2.log 2>&1 &"
