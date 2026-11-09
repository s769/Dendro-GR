#!/bin/bash
#SBATCH -J dualres_res
#SBATCH -p debug-shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A FTA-SUB-Ghattas
#SBATCH -o /home1/08435/srvenkat/Dendro-GR-notes/tier0/sbatch_dualres_%j.out
#SBATCH -e /home1/08435/srvenkat/Dendro-GR-notes/tier0/sbatch_dualres_%j.err
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
bash /home1/08435/srvenkat/Dendro-GR-notes/run_tier15_dualres_resume.sh
