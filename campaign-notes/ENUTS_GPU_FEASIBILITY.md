# ENUTS → GPU feasibility (skim)

## What exists today

- **dendrolib** ships full `ts::ENUTS` + `ENUTSOp` (block async vectors, fine↔coarse time corrections, weighted partition for LTS). See `_deps/dendrolib-src/ODE/include/enuts.h`.
- **BSSN CPU** includes `enuts.h` via `gr.h` but production drivers (`bssngr_main.cpp`, `gr_scaling.cpp`) construct **`ts::ETS<..., BSSNCtx>` only** — uniform global dt.
- **BSSN GPU** (`gr_cuda.cu`, `gr_scaling.cu`) likewise uses **`ts::ETS<..., BSSNCtxGPU>` only**. No `ENUTS<..., BSSNCtxGPU>` instantiation in-tree.

## Gap to drive GPU with LTS

ENUTS assumes:

1. Octree→block decomposition with per-block time levels.
2. Per-block unzipped stage vectors (`BlockAsyncVector` / `BLK_UNZIP` mode).
3. Host-side (or device-mirrored) **block sync + NUTS correction** between levels (`ENUTSOp::{Cfc,Ccf,...}`).
4. Ctx API: `rhs` on a **subset of blocks** at a given time level (or equivalent batching).

`BSSNCtxGPU::rhs` today takes full `DVec` and launches over **all** GPU blocks in one shot (`bssnrhs3_*` with `nblocks`). To plug into ENUTS:

| Piece | Status | Work |
|-------|--------|------|
| ETS↔ENUTS driver swap in `gr_cuda.cu` | missing | Template `ENUTS<DendroScalar, BSSNCtxGPU>` if Ctx meets ENUTS requirements |
| Per-block / per-level RHS batches | missing | Filter or compact `BlockGPU3D` list by time level; launch only active blocks |
| Block-local stage storage on device | missing | Map ENUTS `BlockAsyncVector` ↔ device unzip buffers (or keep stages on host and H2D each substep — kills GPU gains) |
| Ghost/unzip at level interfaces | partial | Current unzip is global CG; LTS needs correct ghosts at **different** times → ENUTS corrections must run (CPU today) or be ported |
| Remesh + LTS repartition | exists on CPU | `WeightedPartitionData` / block regroup tests; must stay consistent with GPU mesh upload |

**Honest estimate:** wiring ENUTS to call the existing monolithic GPU RHS with host-side corrections is a medium project (weeks); a device-resident LTS (stages + corrections on GPU) is a large one. Highest-value first step: **prototype ENUTS+BSSNCtxGPU with host corrections and GPU RHS only on active blocks**, measure sub-cycle efficiency vs launch overhead (ties to C5 CUDA graphs / small-nblocks nsys).

## Why it matters more than RHS 25%

High-q adds fine levels near the small BH; global CFL steps like `2^{Lmax}`. LTS cost ~ `Σ_l N_l 2^l` instead of `N_tot 2^{Lmax}`. For ~+4 levels vs q=1, that is up to ~**16×** step inflation avoided — larger than PT_ASYNC vs stock.

## Recommendation

Treat E1 as the top exploratory track after/alongside FP64 C1–C5. Do not block kernel work on a full LTS port; do schedule a spike: compile-test `ENUTS<...,BSSNCtxGPU>` and list Ctx methods ENUTS needs that GPU Ctx lacks.
