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

### Serial-K ceiling (`DENDRO_UNZIP_K_BENCH`, q8/1GPU, `TIME_END=0.25`, c125)

| K | extra launches | ~ms/launch | ETS (s) |
|--:|---------------:|-----------:|--------:|
| 2 | 1 | ~5.5 | 3.18 |
| 4 | 3 | ~5.5 | 4.66 |
| 8 | 7 | ~5.2 | 7.59 |

Near-linear: serial ensemble unzip ≈ K×. Room for batching only if shared geometry beats gather cost (NCU says gather dominates → expect modest, RHS-like ~1.2–1.5×, not K×).

Raw: `tier0/tier2_frozen_unzip_probe.log`. Hang NCU + TB88 closed: `TIER2_UNZIP.md`.

### KSHARE CTA loop (refEL shared; stride 0 smoke, c125)

`DENDRO_UNZIP_KSHARE=8` vs serial `K_BENCH=8` (probe adds timed work after normal unzip):

| mode | timed work | ~ms total | ~ms/member |
|------|------------|----------:|-----------:|
| serial 7 extra launches | 7× kernel | ~31–38 | ~4.5–5.5 |
| kshare one launch K=8 | 1× CTA loop | ~33–38 | ~4.1–4.7 |

**~10–15% vs serial-K** — matches NCU (gather dominates; sharing `refEL` is small). Not a production ETS lever for K=1. Keep behind env; wire nonzero member strides when ensemble unzip lands.

Raw: `tier0/tier2_kshare_slim.log`.

### Next

1. Ensemble unzip with real `in/out` strides (production path).
2. Coalesced gather staging (true hang-path lever from NCU) — larger change.
3. Queue #6 parked until NCU on RHS spills.
