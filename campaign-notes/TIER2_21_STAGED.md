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

## B1 defer-DD GB200 retest (2026-11-09, c101-021)

`DENDRO_RHS_PT_ASYNC_DEFER=1` via `gen_gauge6_defer_dd.py` → `gauge6_point_async_defer.cpp`
(first+KO early; reload+seconds before K / Gt / Ricci).

| mode | ETS (s) | vs ASYNC | note |
|------|--------:|---------:|------|
| PT_ASYNC | **2.450** | — | prod |
| PT_ASYNC_DEFER | 2.638 | **+7.7%** | **NO-GO** (reload tax > spill win; same as A100) |

Raw: `tier0/tier21_defer_ab.log`. Meshprobe TIME_END=0.25 has no C_HAM samples (ETS-only).

## F32SU C_HAM bit-id (q1 smoke_cons, c101-008)

| mode | ETS (s) | C_HAM samples (5) |
|------|--------:|-------------------|
| ASYNC | 12.646 | 9.84604e-6 … 9.83726e-6 |
| F32SU | 12.566 (−0.6%) | last digits drift ≤1e-11 |

**Not bit-identical** (print-level Δ≤1e-11). Keep F32SU as ETS hedge only — not prod. Raw: `tier0/tier21_f32su_bitid.log`.

## B1 scoped restage (2026-11-09, c101-021)

Section-scoped PT kernel from `bssn_manually_staged.rhs_with_d_computations()` +
`gen_gauge6_pt_scoped.py` (late stages re-load every deriv they need).

| mode | ETS (s) | vs ASYNC | note |
|------|--------:|---------:|------|
| PT_ASYNC | **2.444** | — | prod |
| PT_ASYNC_SCOPED | 5.446 | **+123%** | **NO-GO** — restencil tax dominates |

Raw: `tier0/tier21_scoped_ab.log`. Lesson: staging must cut live ranges **without**
re-running DD stencils on the critical path. Next candidates: (1) outer-scope
`RIJ`/`igt` temps so Ricci runs once then drops gt seconds; (2) reorder-while-hot
(compute seconds while `su` holds the var, use immediately, close scope) with
**no** reload; (3) SymPy CSE reducing DENDRO temp live sets inside the existing
monolithic load order.

## B1 side win — `launch_bounds` (2026-11-09, c101-021)

`eval_rhs3_pt_async`: `__launch_bounds__(343, 3)` → `(343, 1)`.

Smem already limits occupancy to ~1 blk/SM; `minBlocks=3` only tightened the
compiler’s register budget and fed local spills.

| mode | ETS (s) | vs prior ASYNC |
|------|--------:|---------------:|
| ASYNC `lb(343,3)` | ~2.45 | — |
| ASYNC `lb(343,1)` | **2.064–2.084** | **≈ −15–16%** |

NCU lb1 (`tier0/tier21_ncu_lb1.log`, `rhs_pt_async_lb1.ncu-rep`):

| metric | lb3 (B0) | lb1 |
|--------|---------:|----:|
| regs/thread | **56** | **168** |
| smem/block | 17.58 KB | 17.58 KB |
| achieved occ | 17.2% | 17.2% |
| local ld sectors | (spill-heavy) | 131M |

`minBlocks=3` was starving the register file; lb1 lets NVCC keep the working set in regs. Occ still smem-capped. Raw ETS: `tier0/tier21_lb1_ab.log`. **Keep `lb(343,1)` as prod.**

## F32SU under lb1 (2026-11-09)

| mode | ETS (s) | vs ASYNC lb1 |
|------|--------:|-------------:|
| ASYNC lb1 | **2.061** | — |
| F32SU lb1 | 2.358 | **+14%** |

F32SU’s prior −8% edge vs lb3-ASYNC disappears once ASYNC gets the reg budget. **Park F32SU.** Raw: `tier0/tier21_f32su_lb1.log`.

## Status

**B0 done. Defer-DD NO-GO. Scoped-restencil NO-GO. `launch_bounds(343,1)` landed (~−15% ETS).**
Further B1 = live-range/CSE without restencil (still valuable for K/CTA headroom).
B2 ensK baseline: `tier0/tier21_ensK_baseline.md`.