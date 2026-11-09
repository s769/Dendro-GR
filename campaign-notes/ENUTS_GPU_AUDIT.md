# Tier 0.1 — ENUTS vs `BSSNCtxGPU` audit

Gate question: can Dendro’s explicit non-uniform timestepping (ENUTS / LTS) be
brought onto the GPU path used by `bssnSolverCUDA` within a scoped effort?

Sources: `dendrolib` `ODE/include/enuts.h`, `enutsOp.{h,cpp}`, `ctx.h`,
`blkAsync.h`; CPU `BSSNCtx` (`bssnCtx.{h,cpp}`); GPU `BSSNCtxGPU`
(`bssnCtxGPU.cuh/.cu`); drivers `gr_cuda.cu` / `bssngr_main.cpp`.

## What ENUTS needs from a context

| Need | Where used | CPU `BSSNCtx` | GPU `BSSNCtxGPU` |
|------|------------|---------------|------------------|
| Per-block RHS `rhs_blk(in,out,dof,blk,t)` | `BlockAsyncVector` stage advance | **Yes** (host `bssnrhs` / `bssnrhs_sep` on one unzipped block) | **Missing** — only monolithic `rhs(DVec*,…)` over **all** blocks |
| `pre_stage_blk` / `post_stage_blk` | constraint enforce before stage; no-op post | **Yes** (`enforce_bssn_constraints`) | **Missing** |
| `pre_timestep_blk` / `post_timestep_blk` | no-op / post constraints | **Yes** | **Missing** |
| `getBlkTimestepFac(blev,lmin,lmax)` | dt factor `2^{Lmax−blev−offset}` | **Yes** | Declared; **now implemented** (same formula). Still needs `compute_lts_ts_offset` to set `BSSN_LTS_TS_OFFSET` |
| `compute_lts_ts_offset()` | eta/CFL-limited LTS offset | **Yes** | **Missing** |
| Full-mesh `unzip` / `zip` | ENUTS warm-start + final zip | Base `Ctx::unzip/zip` (host mesh) | Same base API exists; GPU production path uses **device** unzip inside `rhs` |
| Per-block unzip/zip (`BlockAsyncVector`, `copyFromUnzip`, `zip`/`zipDG`) | Stage storage + DG ghost fill | Host dendrolib | Host-only; **no device-resident block async vectors** |
| Inter-level **time** interpolation | `ENUTSOp::{Cfc,Ccf,coarser_finer_ut_correction,…}` inside `partial_evolve` / sync | Host CPU loops over dependent elements | **CPU-only**; no CUDA port |
| Partial evolve: RHS only on active time-level subset | `partial_evolve(pt)` skips blocks with `pt % BLK_DT != 0` | Host block loop → `rhs_blk` | GPU would need **compacted block list / launch** per active set |
| Weighted remesh / `grid_transfer` | ENUTS remesh path | CPU + GPU both have remesh/GT | GPU GT rebuilds device mesh — OK if LTS state is re-synced |
| Driver wiring | `ENUTS<T,Ctx>` | Never used in production BSSN drivers (CPU also uses **ETS**) | `gr_cuda.cu` hard-codes **ETS**; `ts_mode==0` argv exists but is unused for ENUTS |

## Architecture mismatch (important)

ENUTS is not “call GPU RHS with a subset of blocks.” It is a **block-async RK**
with:

1. Host `BlockAsyncVector` stages per block.
2. Element-time vectors + ghost exchange of those times.
3. **NUTS correction** (dense small DG stencils) when neighboring elements sit at different times — before each stage sync / unzip into a block.
4. Then `rhs_blk` on that block’s unzipped buffer.

The GPU path today is the opposite shape: zip → **global** device unzip → one
RHS launch over `nblocks` → zip. Ghosts are space-synchronized at one time.

So a naive “ENUTS driver + existing GPU `rhs`” either:

- **A.** Keeps ENUTS host logic and calls GPU only inside `rhs_blk` for one (or a
  batch of) blocks — pays H2D/D2H every substep; small-nblocks launch overhead
  dominates (Tier 0.3), or
- **B.** Ports block-async state + corrections + partial launches to device —
  real LTS on GPU — large effort.

## Gap list with effort

| ID | Gap | Effort | Notes |
|----|-----|--------|-------|
| G1 | Implement GPU Ctx LTS hooks: `rhs_blk`, `pre/post_stage_blk`, `pre/post_timestep_blk`, `compute_lts_ts_offset` | **S** (2–4 d) | Can wrap existing device RHS for a single block ID, or batch IDs into a compacted launch |
| G2 | Wire `ts::ENUTS<DendroScalar,BSSNCtxGPU>` in `gr_cuda.cu` (`ts_mode==0`) | **S** (1–2 d) | Compile/link + remesh sync; will not be fast until G3–G5 |
| G3 | Active-set block list compaction + GPU RHS launch over subset | **M** (1–2 wk) | Required for any speedup; ties to Tier 1.3 / CUDA graphs |
| G4 | Device-resident RK stage buffers for active blocks (avoid H2D each substep) | **M–L** (2–4 wk) | Without this, A is a no-go on throughput |
| G5 | Port or bypass `ENUTSOp` time corrections on GPU (or restrict to level-synchronous batches with host correction) | **L** (1–2 mo for full port; **M** for hybrid host-correct) | Correctness-critical; growing cost at fine/coarse interfaces |
| G6 | Partial unzip/zip for dependent element sets (today: full CG unzip then `copyFromUnzip`) | **M** (2–3 wk) | ENUTS already does element-local DG gather; GPU lacks equivalent |
| G7 | Bit-repro + constraint atomics under LTS (Tier 1.4) | **S–M** | Order-fixed reductions |
| G8 | Ensemble×LTS interaction (shared mesh, K members, fine levels tiny) | **M** | Only after Tier 1.3; motivates K-members-per-CTA |

**Scoped spike (recommended before go):** G1+G2+minimal G3 (host ENUTS, GPU RHS on compacted active set, stages still host) → measure subcycle efficiency vs ETS on a q≈4 mesh. Stop if active sets are ≪64 blocks and kernel time ≪ launch+PCIe.

## Already provided by GPU context

- Full-mesh device RHS (PT_ASYNC etc.) and PreferShared occupancy path.
- Device mesh (`MeshGPU` / `BlockGPU3D`) with `m_level` on `TNode3d`.
- `unzip`/`zip` via Ctx base (host) and device unzip inside `rhs`.
- Remesh + `grid_transfer` that rebuilds the device mesh.
- `getBlkTimestepFac` (implemented to match CPU).

## CPU-only today

- Entire ENUTS control plane (`partial_evolve`, blk sync, NUTS corrections).
- `BlockAsyncVector` / BLK_UNZIP stage storage.
- `rhs_blk` and block-level constraint hooks.
- `compute_lts_ts_offset`.
- Production BSSN drivers: **ETS only** on both CPU and GPU.

## Go / no-go (pre-0.2/0.3)

**Conditional go** only for a **hybrid** LTS (host ENUTS + GPU batched RHS), and only if Tier 0.2 shows ≥~5× theoretical GTS/LTS work ceiling **and** Tier 0.3 shows that sub-100-block launches can be amortized (graphs / ensemble K / fused zip).

**No-go on full device-resident ENUTS** for this campaign’s effort budget: G4+G5 alone are multi-month and sit beside Tier 1 ensemble work. If 0.2 ceiling is only ~2–3×, **kill GPU LTS** and re-cost SVN around hybridization length + resolution ladder (Tier 3.2–3.3).

Numbers for the final gate: see `TIER0_LTS_GATE.md` (filled after 0.2/0.3 runs).
