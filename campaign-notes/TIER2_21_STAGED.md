# Tier 2.1 — staged algebra

## GB200 NCU (`eval_rhs3_pt_async`, q8, 2026-11-08, c106)

Report: `tier0/rhs_pt_async_occ.ncu-rep` · CSV: `tier0/tier21_ncu_rhs.csv` · log: `tier0/tier21_ncu_rhs.log`

| metric | value |
|--------|------:|
| Registers / thread | **56** |
| Dynamic smem / block | **17.58 KB** |
| Theoretical / achieved occupancy | **17.2% / 17.2%** |
| Occupancy limiter (NCU) | **shared memory** |
| Local mem spilling requests | **109M (100% overhead)** |
| Local mem share of L1TEX sectors | **84.3%** |
| Compute (SM) / Memory / DRAM throughput | 13.5% / 38.7% / 23.2% |
| FP64 peak (roofline note) | ~**7%** |
| Eligible warps / scheduler | **0.13** |
| Warp cycles / issued insn | 24.9 (L1TEX scoreboard ~53%) |

**Gate:** spills + local-mem dominance → **unpark 2.1**. Staging is required before K/CTA interleave (BATCHING.md): 56 regs × K and 17.6 KB×K do not leave headroom; cutting live ranges / killing local spills is the lever. Occupancy is also smem-bound → K/CTA must keep **one var tile** in smem (user note), not K tiles.

## Prior art (A100)

`gen_gauge6_point.py` deferred DD seconds until before K: bit-identical but **slower** on A100 (0.157 vs 0.137) — 11-var gmem reload cost more than the live-range win. Unstaged stayed default.

GB200 result changes the priority: local mem is crushing L1TEX; a reload that removes spills can still win even if A100 lost.

## Plan

1. ~~NCU~~ **done** (above).
2. Identify local-mem culprits (large autos / high live range in `gauge6_point_async.cpp`).
3. Regen staged algebra (`CodeGen/bssn_manually_staged.py` / `--staged_type`) behind env; A/B ETS + bit-id vs unstaged on c106.
4. Only then B2: K/CTA interleave speedup curve.

## RHS path A/B (q8, TIME_END=0.25, 1 GPU, c106)

| mode | ETS (s) | vs ASYNC | note |
|------|--------:|---------:|------|
| **PT_ASYNC** (prod) | **2.449** | — | smem 17.6 KB, occ 17% |
| PT_GMEM | 3.697 | **+51%** | 0 smem → **NO-GO ETS** (L1/L2 miss tax > occ win) |
| PT_F32SU | 2.243 | **−8.4%** | float su tile; **needs C_HAM bit-id** before prod |

Raw: `tier0/tier21_rhs_ab.log`. GMEM confirms: don’t chase occupancy by dropping the shared stencil tile. F32SU is a candidate smem shrink, not algebra staging.

## Status

**B0 done.** B1 = true live-range / CSE staging inside PT_ASYNC (not GMEM, not WS-HBM). F32SU bit-id is a side check.