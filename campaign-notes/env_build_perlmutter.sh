#!/usr/bin/env bash
# Source tree: $HOME/Dendro-GR
# Build + profiles: $SCRATCH
set -euo pipefail
module load PrgEnv-gnu
module load cmake/3.30.2
module load cudatoolkit/13.2
export LD_LIBRARY_PATH="${CUDATOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC=cc CXX=CC FC=ftn
export CUDACXX="$(which nvcc)"
export CUDAHOSTCXX="$(which g++)"
export GSL_ROOT_DIR=/usr
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DENDRO_ROOT="${DENDRO_ROOT:-$HOME/Dendro-GR}"
export DENDRO_BUILD="${DENDRO_BUILD:-/pscratch/sd/s/srvenkat/Dendro-GR-build}"
export DENDRO_GPUOPT="${DENDRO_GPUOPT:-/pscratch/sd/s/srvenkat/Dendro-GR-gpuopt}"
echo "DENDRO_ROOT=$DENDRO_ROOT"
echo "DENDRO_BUILD=$DENDRO_BUILD"
