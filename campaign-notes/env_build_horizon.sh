#!/usr/bin/env bash
# Horizon GB200 build environment for Dendro-GR
# Usage: source ~/Dendro-GR-notes/env_build_horizon.sh
set -euo pipefail
module purge
module load cmake/4.4.0 gcc/15.3.0 cuda/13.2 ucc/1.9.0 openmpi/5.0.11

export CC=mpicc CXX=mpicxx FC=mpifort
export CUDACXX="$(which nvcc)"
export CUDAHOSTCXX="$(which g++)"
export GSL_ROOT_DIR="${GSL_ROOT_DIR:-$HOME/software/gsl-2.8}"
# gcc libisl (nvcc host preprocess) + NVPL blas/lapack (login image has no flexiblas)
export NVPL_LIB="${NVPL_LIB:-/home1/apps/nvidia/Linux_aarch64/26.9/compilers/lib}"
export LD_LIBRARY_PATH="/opt/apps/gcc/15.3.0/lib:${GSL_ROOT_DIR}/lib:${NVPL_LIB}:${LD_LIBRARY_PATH:-}"

export DENDRO_ROOT="${DENDRO_ROOT:-$HOME/Dendro-GR}"
export DENDRO_BUILD="${DENDRO_BUILD:-$HOME/Dendro-GR-build}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Do not default-pin CUDA_VISIBLE_DEVICES — it silently collapses multi-GPU
# mpirun jobs onto GPU0. Set explicitly only for single-GPU benches.

MPI_INC="$(mpicc -showme:compile 2>/dev/null | tr ' ' '\n' | awk '/^-I/{sub(/^-I/,""); print; exit}')"
MPI_INC="${MPI_INC:-/opt/apps/gcc15/cuda13/openmpi/5.0.11/include}"
BLASLIB="${NVPL_LIB}/libblas.so"
LAPACKLIB="${NVPL_LIB}/liblapack.so"

dendro_cmake() {
  mkdir -p "$DENDRO_BUILD"
  cmake -S "$DENDRO_ROOT" -B "$DENDRO_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=ON \
    -DMPIX_CUDA_AWARE_SUPPORT=ON \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DCPU_ARCH=native \
    -DGSL_ROOT_DIR="$GSL_ROOT_DIR" \
    -DCMAKE_PREFIX_PATH="$GSL_ROOT_DIR" \
    -DBLAS_LIBRARIES="$BLASLIB" \
    -DLAPACK_LIBRARIES="$LAPACKLIB" \
    -DCMAKE_CUDA_FLAGS="-I${MPI_INC}"
}

dendro_build_rhsgpu() {
  cmake --build "$DENDRO_BUILD" -j"${DENDRO_BUILD_JOBS:-16}" --target rhsgpu_test
}

echo "DENDRO_ROOT=$DENDRO_ROOT"
echo "DENDRO_BUILD=$DENDRO_BUILD"
echo "nvcc=$(which nvcc)  mpicxx=$(which mpicxx)"
echo "Helpers: dendro_cmake   dendro_build_rhsgpu"
