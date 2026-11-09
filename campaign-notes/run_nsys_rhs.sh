#!/bin/bash
set -euo pipefail
module purge 2>/dev/null || true
module load cmake/4.4.0 gcc/15.3.0 cuda/13.2 ucc/1.9.0 openmpi/5.0.11 2>/dev/null || true
export GSL_ROOT_DIR=$HOME/software/gsl-2.8
export LD_LIBRARY_PATH=$GSL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PATH=/home1/apps/nvidia/Linux_aarch64/26.9/compilers/bin:$PATH
NSYS=$(command -v nsys)
BIN=$HOME/Dendro-GR-build/BSSN_GR
OUT=$HOME/Dendro-GR-notes/nsys
mkdir -p "$OUT"
TAG=${1:-ptasync}
MODE=${2:-ptasync}   # stock|ptasync|ptdb|pt
NBLOCKS=${3:-512}
NREPS=${4:-5}
BFLAG=${5:-0}
cd "$BIN"
case "$MODE" in
  stock) ENV_CMD=() ;;
  pt) ENV_CMD=(env DENDRO_TIME_PT=1) ;;
  ptasync) ENV_CMD=(env DENDRO_TIME_PT_ASYNC=1) ;;
  ptdb) ENV_CMD=(env DENDRO_TIME_PT_DB=1) ;;
  *) echo bad mode; exit 1 ;;
esac
"${ENV_CMD[@]}" ./rhsgpu_test "$NBLOCKS" 1 "$BFLAG" >/dev/null 2>&1 || true
REP="$OUT/${TAG}_b${BFLAG}_$(hostname -s)"
rm -f "${REP}.nsys-rep" "${REP}.sqlite"
echo "Profiling $TAG mode=$MODE on $(hostname) GPU=$CUDA_VISIBLE_DEVICES"
$NSYS profile -t cuda,nvtx -s none -o "$REP" --force-overwrite=true \
  "${ENV_CMD[@]}" ./rhsgpu_test "$NBLOCKS" "$NREPS" "$BFLAG" \
  > "${REP}.log" 2>&1
$NSYS stats --report cuda_gpu_kern_sum "${REP}.nsys-rep" 2>/dev/null | head -50 | tee "${REP}_kern.txt"
echo DONE_NSYS_$TAG
