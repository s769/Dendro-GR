#!/usr/bin/env bash
# wait for the fine fiducial (CAHD) to finish, stop the fiducial node's script, then run replays on node of job 88787
D=$HOME/Dendro-GR/campaign-notes
OUT=$HOME/Dendro-GR-runs/tier15_wave_fine_T26_cahd
until sed 's/\x1b\[[0-9;]*m//g' $OUT/fid/run.log 2>/dev/null | grep -q "ETS time"; do sleep 20; done
echo "$(date -Is) fine fiducial done" >> $D/tier0/fine_replays.log
# stop the fiducial node's script before it starts its (doomed) replays
ssh -o StrictHostKeyChecking=no c103-027 "pkill -u $USER -f run_tier15_waveform_fd.sh; pkill -u $USER -x bssnSolverCUDA" 2>/dev/null
until [ -n "$(squeue -j 88787 -h -t R -o %N)" ]; do sleep 20; done
N=$(squeue -j 88787 -h -t R -o %N); sleep 20
echo "$(date -Is) launching replays on $N" >> $D/tier0/fine_replays.log
R='[10.0, 12.0, 15.0, 18.0, 20.0, 25.0]'
ssh -o StrictHostKeyChecking=no $N "nohup env SKIP_FID=1 RES=fine TEND=26 RFREQ=20 OUTTAG=_cahd PAR_OVR='BSSN_CAHD_C=0.06' RADII='$R' bash $D/run_tier15_waveform_fd.sh > $D/tier0/wave_fine_cahd_rep.log 2>&1 &"
