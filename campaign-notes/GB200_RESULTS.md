# Horizon GB200 optimization results

Updated 2026-11-07. Microbench: `rhsgpu_test 2048 20 0` (interior blocks).

## RHS microbench (single GPU)

| Path | Time (s) | vs stock | Notes |
|------|--------:|---------:|-------|
| stock `eval_rhs3` | ~0.106–0.110 | 1.0× | PreferShared, 52KB smem |
| PT_ASYNC | ~0.061–0.067 | ~1.6–1.8× | bit-id interior; hybrid faces |
| PT_DB | ~0.063 | ~1.7× | 2×su prefetch; rarely beats ASYNC |
| F32C | ~0.056–0.061 | ~1.8–2.0× | centered float Du/DDu; PreferShared |
| F32 (raw) | ~0.054 | ~2.0× | larger error (~1e-4 α) |
| PT_KOF32 | ~0.072 | slower | |
| PT_GMEM | ~0.119 | slower | 48% occ but HBM-bound |
| **PT_F32SU** | **~0.052** | **~2.0–2.1×** | **PT fused + float su; PreferShared — current best** (syncs restored for determinism; was racing ~0.050) |

## Accuracy (smoke cons vs stock)

| Path | max\|ΔC_HAM\| | rel ΔC_HAM |
|------|-------------:|-----------:|
| F32C | 2.8e-11 | **2.8e-6** |
| PT_F32SU | 2.8e-11 | **2.8e-6** |
| PT_* FP64 (hybrid) | ~0 (bit-id interior) | — |

**Production default: `DENDRO_RHS_PT_ASYNC=1` (FP64).** PT_F32SU is a **hedge only** (~25% faster than ASYNC) — keep validated (A3 on idle GPUs); do not expand FP32 as the main line. Kernel is latency/structure-bound (~10% FP64 peak), so precision is the wrong lever on GB200.

**Tensor cores:** closed for BSSN RHS (stencil K=7/13; see `HANDOFF_NEXT.md`).  

See **`HANDOFF_NEXT.md`** for FP64 structural + ENUTS/8th-order roadmap.

## Nsight Compute (GB200, CC 10.0)

**Stock / PT_ASYNC + PreferL1:** 56 regs; PT_ASYNC **smem-limited to 1 blk/SM** → 17% occ.
**F32C / PT_F32SU + PreferShared:** register-limited 3 blk/SM → **~45–51% occupancy** (smem allows 8–23). PreferL1 on these paths is a large regression (carveout too small).
**PT_GMEM:** 48% occ but ~2× slower (HBM stencil traffic).

Partial-wave / 343≠multiple-of-32: ≤2.5% estimated.

## Multi-GPU (1 node × 4×GB200)

Independent `rhsgpu_test` per GPU: ~0.017s each @1024×10 — no cross-talk on microbench.

## Env knobs

```
DENDRO_RHS_PT_ASYNC=1   # PRODUCTION — FP64 bit-id interior + stock faces
DENDRO_RHS_PT_F32SU=1   # HEDGE only — faster; finish A3 before trusting at high-q
DENDRO_RHS_F32C=1       # older mixed-precision experiment
```
