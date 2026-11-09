# Handoff: Dendro-GR GPU — production line (revised)

**Production default: `DENDRO_RHS_PT_ASYNC=1` (FP64, bit-identical interior + stock faces, ~1.7× stock).**  
**Hedge (not default): `DENDRO_RHS_PT_F32SU=1` (~2.1×) — keep in tree, finish A3 on idle GPUs, do not expand FP32 work.**  
**Dropped: tensor-core formulations for BSSN RHS** (see § Tensor cores).

Rationale: FP32 buys ~25% over PT_ASYNC but carries open-ended high-q validation risk. The kernel sits at ~10% FP64 peak / <20% HBM — latency/structure-bound, not precision-bound. Staged algebra, zip fusion, and 19³ blocks are bit-identical and should recover much of that 25%. High-q cost is dominated by **step count** (global dt from h_min); algorithmic levers (LTS / higher order) multiply that, kernel opts only scale it linearly.

Report numbers in the same format as `GB200_RESULTS.md`. Keep paths env-selectable.

---

## Priority order (do these)

### 1. FP64 structural (bit-identical)

| ID | Item | Notes |
|----|------|-------|
| C1 | **Staged algebra** | NCU on PT_F32SU/PT_ASYNC: L1TEX long-scoreboard dominated, 56 regs, 0 spills. Stage gauge+χ+g̃ (1st derivs) then Ricci/Ã/K/Γ̃ (2nd). Target ≤46 regs → 4 CTA/SM. Measure spills, occ, time on **PT_ASYNC**. |
| C3 | **Fuse zip into RHS** | Five-minute check first: GPU zip = interior-node copy? Then scatter Fu→nodal. Bit-id vs zip(RHS) on PT_ASYNC. |
| C4 | **19³ blocks** | Dump `m_sz` histogram for q=4/q=8 CPU mesh. Halo 84%→68%. FP64 19³ tile = 54.9 KB — fits PreferShared. |
| C5 | **Launch hygiene** | One-sided FD in fast path; compact face/interior lists; CUDA graphs for unzip→RHS→zip→update; nsys at 128/512 blocks (LTS-relevant). |

### 2. Real timing

| ID | Item |
|----|------|
| D | **q=4 evolve ~100 M on 4 GPUs**, `DENDRO_RHS_PT_ASYNC=1`, `__PROFILE_CTX__` on. Replace extrapolated BBH table with measured RHS/unzip/zip/ghost/remesh split. |

### 3. High-leverage investigations (exploratory)

| ID | Item | Why |
|----|------|-----|
| E1 | **ENUTS → GPU** | Dendro already `#include "enuts.h"` in BSSN. Local time stepping recovers most of the ~16× step inflation from +4 fine levels at high q — larger than all kernel gains combined. Feasibility: can ENUTS drive `BSSNCtxGPU`, or what wiring is missing? |
| E2 | **8th-order (pw=4)** | Cost ~h⁻⁴. Coarser h_min at fixed waveform error can beat every RHS tweak; also widens any future FP32 safety margin. Convergence study; codegen already parameterized on `pw`. |

### 4. Hedge only (idle GPU)

| ID | Item |
|----|------|
| A3 | High-q (~q=8/16 h_min) ~20 M stress for PT_F32SU vs PT_ASYNC. Pass/fail as before. **Do not block FP64 work on this.** |
| A0 | Done — F32SU bit-identical after restoring post-stencil syncs. |

Deprioritized / cancelled as main line: A1/A2 full FP32 campaigns, A4 affine/hybrid float tiles, C2 all-tiles-resident (barrier-oriented; B pointed at scoreboard/live ranges instead).

---

## Tensor cores: closed (do not re-derive)

A tensor core pays when reduction length **K** is large vs data moved. BSSN FD has **K = 7** (6th-order) or **K = 13** (pencil); batching vars/blocks grows **N**, not **K**. Dense MMA formulations do ≥2× the flops of the stencil; on GB200 FP64 tensor ≈ FP64 vector rate → net loss. TF32 (10-bit mantissa) is far too inaccurate for derivatives. Compensated / Ozaki-style FP64 emulation amortizes at K ~ thousands, not K = 13. Unzip/remesh 7×7 tensor-product interp is also K = 7 and memory-bound.

Keep this paragraph so tensor cores are not reopened for the RHS kernel. (Blackwell Ultra-class FP64 cuts make the **FP32 hedge** more valuable on future silicon — another reason to keep PT_F32SU validated, not to default it on GB200.)

---

## Current microbench (GB200, `2048×20`, interior)

| Path | Time | Role |
|------|-----:|------|
| stock | ~0.106–0.110 s | baseline |
| **PT_ASYNC** | **~0.061–0.067 s** | **production** |
| PT_F32SU | ~0.052 s | hedge only |

---

## Log

| Item | Status | Artifact |
|------|--------|----------|
| Strategy revision | **this doc** | FP64 default |
| B stalls | done | `ncu/ptf32su_stalls_summary.txt` — apply same NCU to PT_ASYNC |
| A0 F32SU determinism | pass | sync fix in `gen_gauge6_point_f32su.py` |
| val20 q1 ASYNC/F32SU | **done (hedge)** | 21 samples; ‖C_HAM‖ ratio F32SU/ASYNC ∈ [1.000, 1.0005], max\|Δ\|~1.5e-8, no secular drift at this (q1/smoke) resolution. Wall ~1752s vs ~1768s (unzip/remesh-dominated). **A3 at high-q h_min still required before trusting F32SU.** |
| C1 staged FP64 | next | |
| E1 ENUTS skim | next | |
| D q=4 timed | after C progress | |
