#!/usr/bin/env bash
cd $HOME/Dendro-GR/campaign-notes
for f in 4 100000; do
  RES=coarse TEND=8 RFREQ=20 GWFREQ=$f OUTTAG=_gwf$f RADII='[10.0,12.0,15.0,18.0,20.0,25.0]' EPSS='' bash ./run_tier15_waveform_fd.sh 2>&1 | grep -E "gpu0 wall"
done
echo GWF_DONE
