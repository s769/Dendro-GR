# B3 — Pack K ghosts + one RHS launch

## Goal

For frozen-mesh ensemble (K≈4–8): one ghost exchange and one RHS launch covering all members. Dominates at fine LTS (few blocks) where launch/msg overhead wins over compute.

## What’s already in

| piece | status |
|-------|--------|
| One RHS launch, `grid.y=K` | **done** (Tier 1.3 v1; `bssnrhs3_pt_async(..., K)`) |
| K/CTA share (v2) | **no-go** at 64bl vs v1 |
| Solver-level multi-member unzip/ghosts | **not wired** — only `rhsgpu_test` microbench |
| CUDA-aware single-member ghosts | **done** |

## Gap

Production `bssnCtxGPU` still runs K=1. Ensemble packing needs:

1. **Layout** — leading member stride `BSSN_NUM_VARS * szpdof_uz` (same as RHS ensK).
2. **Ghost pack** — `__write_sb_cg__` / `read_from_ghost_cg_*` looped over K *or* a single kernel that packs K×sendbuf with member stride (one MPI payload per neighbor).
3. **Unzip** — geometry once (frozen mesh); field gather ×K (or fused gather with member loop).
4. **Measure** — at 32/64/128 blocks: wall for (serial K ghosts+RHS) vs (packed ghosts + one ensK RHS).

## Measure harness (GPU)

When a node is up:

```bash
# Micro: RHS already fused; ghost pack A/B is the new part
# Full: short q8 TIME_END with DENDRO_ENSEMBLE_K once ctx wired
```

First GPU deliverable: **nsys / ETS at fine nblocks** comparing K serial ghost rounds vs one packed exchange (even as a host-loop of device pack kernels into a contiguous send buffer — proves bandwidth packing before fusing the pack kernel).

## Non-goals

- Reopening async_comm_K>1 for single-member (already hurts).
- K/CTA v2.
- FP32.

## Status

Design note 2026-11-09. Updated same day on c126-026 (87314):

- **RHS fused ensK at fine nblocks:** measured (`tier0/b3_fine_ensk.md`) — 32bl/K=8 → **2.89×** vs serialK.
- **Ghost pack kernels:** `__write_sb_cg_ensK__` / `__read_rb_cg_ensK__` + `MeshGPU::read_from_ghost_cg_ensK_{begin,end}` landed in dendrolib `_deps` (`mesh_gpu.cuh` / `mesh_gpu.h`). Layout: per-neighbor contiguous `K×` so one Isend/Irecv. Ctx buffers must be sized `K * dof * {send,recv}BSz`. **Not yet called from bssnCtxGPU** — needs multi-GPU measure next.

## 4-GPU measure (2026-11-09, c126-026)

q8 meshprobe `DENDRO_B3_PACK_BENCH`: packed **slower** than serial (K=2/4/8 → 0.84/0.60/0.53×). Expected — q8 is not fine-LTS. Remeasure when a few-block mesh is available. See `tier0/b3_pack_bench_4gpu.md`.

## q1 remeasure (2026-11-09)

Same NO-GO: K=2/4/8 → 0.86/0.66/0.54× (`b3_pack_q1_4gpu.log`). Mesh still O(10³) elements — pack needs few-block fine LTS. **Park ctx pack wire**; keep ensK RHS for fine nblocks.

