# Tier 2 / queue #5 — frozen-mesh unzip reuse (ensemble)

## Why

Hang-path NCU (`__unzip_cg1__2`): wall is **uncoalesced CG gather/scatter**, not CTA size.
Single-member micro-opts (split, TB88, batch-sync, async_k) are closed no-go/neutral.

On a **frozen mesh**, element→node maps, hang flags, and `refEL` are invariant across:
- RK stages within a step (state changes; maps do not)
- Ensemble members K (Tier 1.3) sharing one mesh

RHS already reuses block geometry via `grid.y=K` / `kshare`. Unzip still pays full hang prolong **per member × per var**.

## Target

1. **K-batch unzip**: one CTA per (element[, var]) loops or grids K members sharing `refEL` + hang prolong workspace; gather/scatter per member.
2. Measure: unzip ms for K∈{1,2,4,8} serial vs batched on frozen q1/q8 block list.
3. Success: K=8 unzip ≪ 8× K=1 (target ≤ ~3–4×; geometry amortize).

## Non-goals

- Changing hang prolong arithmetic (FP64).
- Remesh-path MeshGPU rebuild (already skipped under replay for fiducial flags).

## Status

- Design only. Next: microbench in `rhsgpu_test` or a tiny `unzip_k_bench` launching current kernel K times vs a K-loop prototype.
- Hang NCU + TB88 closed: see `TIER2_UNZIP.md`.
