# Tier 1.4 — Bit reproducibility (min-det / reductions)

## Change

`cuda_bssn_enforce_evar_cons` no longer uses `atomicCAS` for the pre-enforcement `min det(gt)` diagnostic. Each CTA block-reduces its local min (warp shuffle + shared), writes `block_mins[blockIdx.x]`, and `BSSNCtxGPU::post_timestep` folds `block_mins[0..nblk)` in ascending index into the running device min.

## Validation (c101-014, job 86504)

Dual identical `q1.smoke_cons` runs (`DENDRO_RHS_PT_ASYNC=1`):

| Field | Result |
|-------|--------|
| `MIN_DET_GT` column | **bit-identical** all steps |
| Constraint L2 (`C_HAM`, momenta, …) | `max\|Δ\| ≈ 1.5e-18` (FP noise in norm reductions) |
| `U_ALPHA` (min,max) log lines | match |

So the former atomic is fixed. Residual non-bit-id is in **norm / L2 reductions**, not enforcement or min-det. Full order-fixed norms still open if strict bitwise ensemble slots are required end-to-end.

Mesh record/replay still works after the change (remesh at steps 2, 4).

## Related

Ensemble RHS slots already bit-match K=1 (`TIER1_ENSEMBLE.md`). Init WAMR skipped under `DENDRO_MESH_REPLAY` (`TIER1_MESH_REPLAY.md`).
