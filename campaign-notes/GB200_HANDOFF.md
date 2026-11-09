# Dendro-GR GPU RHS — handoff for GB200 / Blackwell

**Date:** 2026-11-06 (Perlmutter A100 results)  
**Keep:** point-wise fused kernel + PreferL1  
**A100 microbench:** `rhsgpu_test 2048 20 0` → baseline **0.171 s** → PT **0.137 s** (**+20%**, bit-identical)

Full experiment table: `OPT_EXPERIMENTS.md` (same directory).

---

## What to transfer

Tarball (preferred):

```text
/pscratch/sd/s/srvenkat/Dendro-GR-gpuopt/transfer/dendro-gr-gpuopt-pt.tgz
```

Contents: patch of `Dendro-GR` CUDA RHS sources (point-wise path + F32 variants + notes + helper scripts).  
Apply on top of a normal Dendro-GR checkout (same `master` lineage).

Or rsync the live tree:

```bash
# From Perlmutter (or any host that can see this HOME):
rsync -aL --exclude '.git' --exclude '.cursor' \
  /global/homes/s/srvenkat/Dendro-GR/ \
  USER@GB200HOST:~/Dendro-GR/
```

Minimal file set if you prefer a manual copy:

| Path under `Dendro-GR/` | Role |
|-------------------------|------|
| `BSSN_GR/cuda/include/derivs_point.cuh` | Point + fused stencils, cp.async helpers |
| `BSSN_GR/cuda/include/derivs_f32.cuh` | FP32 + **centered** load |
| `BSSN_GR/cuda/include/derivs_f2d.cuh` | Hybrid f2d (experimental) |
| `BSSN_GR/cuda/include/bssn_kernels.cuh` | `eval_rhs3_pt` / `_f32` / `_f32c` / … |
| `BSSN_GR/cuda/scripts/gen_gauge6_point.py` | Codegen |
| `BSSN_GR/cuda/scripts/gauge6_point.cpp` | Generated algebra+derivs |
| `BSSN_GR/cuda/scripts/gauge6_point_async.cpp` | cp.async issue+wait variant |
| `BSSN_GR/cuda/scripts/gauge6_f32.cpp` | F32 Du/DDu |
| `BSSN_GR/cuda/scripts/gauge6_f32c.cpp` | Centered F32 |
| `BSSN_GR/cuda/src/rhs_test.cu` | `DENDRO_TIME_*` timers |
| `BSSN_GR/cuda/src/bssnCtxGPU.cu` | `DENDRO_RHS_PT` / `DENDRO_RHS_F32` |

---

## Build (adapt modules for your GB200 box)

Perlmutter reference (`env_build.sh`):

```bash
module load PrgEnv-gnu cmake/3.30.2 cudatoolkit/13.2   # or site equivalent
export CC=cc CXX=CC FC=ftn
export CUDACXX=$(which nvcc) CUDAHOSTCXX=$(which g++)
```

**Critical for Blackwell:** set the CUDA arch to **sm_100** (or whatever `nvidia-smi` / site docs say for GB200). On A100 we used `sm_80`. Example CMake hint:

```bash
cmake -S "$DENDRO_ROOT" -B "$DENDRO_BUILD" \
  -DCMAKE_CUDA_ARCHITECTURES=100 \
  ...your usual Dendro flags...
cmake --build "$DENDRO_BUILD" -j8 --target rhsgpu_test
# optional: bssnSolverCUDA
```

If the project hardcodes `compute_80`, search/replace or pass the arch flag your CMake already supports.

---

## First commands on GB200

```bash
# 1) Microbench — baseline vs point-wise
./BSSN_GR/rhsgpu_test 2048 20 0
DENDRO_TIME_PT=1 ./BSSN_GR/rhsgpu_test 2048 20 0
# Expect: pt_gpu_time printed; l_inf ~1e-14..0 (same as CPU)

# 2) Optional variants
DENDRO_TIME_F32C=1 ./BSSN_GR/rhsgpu_test 2048 20 0   # centered float scratch
DENDRO_TIME_PT_KOF32=1 ./BSSN_GR/rhsgpu_test 2048 20 0
DENDRO_TIME_PT_ASYNC=1 ./BSSN_GR/rhsgpu_test 2048 20 0  # issue+wait only; slower on A100

# 3) Production RHS path (solver)
DENDRO_RHS_PT=1 ./bssnSolverCUDA ...your.par.toml...
# Revert: unset DENDRO_RHS_PT (stock eval_rhs3)
```

`rhsgpu_test` args: `<nblocks> <nReps> <bflag>` — use `bflag=0` (interior).

---

## Env cheat sheet

| Env | Effect |
|-----|--------|
| `DENDRO_TIME_PT` | Time + verify `bssnrhs3_pt` in `rhsgpu_test` |
| `DENDRO_TIME_PT_ASYNC` | cp.async load (no overlap yet) |
| `DENDRO_TIME_PT_KOF32` | PT + KO in FP32 |
| `DENDRO_TIME_F32` / `F32C` / `F2D` / `WS` / `RHS2` / `RHS4` | Other experiments |
| `DENDRO_PT_PREFER_SHARED=1` | PreferShared for PT (A100: PreferL1 is faster) |
| `DENDRO_RHS_PT=1` | Solver uses point-wise RHS |
| `DENDRO_RHS_F32=1` | Solver uses F32 Du/DDu (C_HAM risk) |

---

## A100 findings (so you don’t re-learn them)

1. **Smem-bound hypothesis holds** for stock `eval_rhs3` (210 full-block passes). PreferShared wins there; HBM workspaces lose; eliding barriers does nothing.
2. **Point-wise + fused** (items 1–2) cuts Du/DDu traffic; with only ~18 KB smem, **PreferL1** wins (+20%).
3. **lb(343,4)** on PT increased spills → slower. Keep **lb(343,3)**.
4. **Algebra staging** (defer seconds until Ricci) is correct but **slower** (reload 11 fields).
5. **cp.async issue+wait** without overlap is **slower**; true double-buffer (prefetch v+1 while computing v) not finished.
6. **Centered F32** `(u−u_c)` ≈ F32 speed, ~100–500× smaller RHS error than raw float cast. Still not bit-identical — watch C_HAM on evolve.
7. Skip FP64 tensor cores for this kernel on Blackwell; vector FP64 + smem/L1 restructuring is the path.
8. GB200 has more smem/SM (~228 KB usable class) → room for **19³ blocks**, multi-`su` residency, or DSMEM cluster prototypes.

---

## Suggested GB200 order of work

1. Rebuild `rhsgpu_test` for sm_100; confirm PT still bit-identical and measure vs baseline (expect >1.2× from hardware alone on stock; PT should stack).
2. `DENDRO_RHS_PT=1` short evolve; compare C_HAM / waveforms to stock.
3. Implement **overlapped** double-buffer (`su_A`/`su_B` + `cp.async`).
4. Histogram real `m_sz` from a production mesh; try 19³ if locally uniform.
5. Optional: F32C evolve vs PT for mixed-precision roadmap (Blackwell Ultra FP64 cuts).

---

## Regenerating point-wise algebra

```bash
cd Dendro-GR/BSSN_GR/cuda/scripts
python3 gen_gauge6_point.py   # writes gauge6_point.cpp
# then rebuild rhsgpu_test / solver
```
