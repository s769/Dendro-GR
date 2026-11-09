# Tier 1.1 — Frozen-mesh remesh record / replay

## API (env)

| Env | Effect |
|-----|--------|
| `DENDRO_MESH_RECORD=<dir>` | Before each remesh, write local refinement flags; after remesh, write oct snapshot |
| `DENDRO_MESH_REPLAY=<dir>` | Skip WAMR; at recorded steps apply flags + `remesh_and_gridtransfer` |

Files under `<dir>`:

- `mesh_history.manifest` — `step time`
- `mesh_flags_<step>_<rank>_<npes>.bin` — `uint32 n` + flags (`OCT_SPLIT` / `COARSE` / `NO_CHANGE`)
- `mesh_oct_*.oct` — diagnostic post-remesh octants (not used for replay)

Replay uses **flags + ReMesh**, not orphan octree rebuild (IGT aborts otherwise).

## Validation (c101-014, q1 smoke)

Fiducial record → replay:

| Check | Result |
|-------|--------|
| Remesh steps applied | 2, 4 (same as record) |
| `U_ALPHA` (min,max) lines | match |
| `max\|ΔC_HAM\|` over 5 samples | **1.04e-18** (FP noise; first sample exact) |

Pass for practical bit-repro. Stricter bitwise evar dump can be added later if needed.

## Note for non-fiducial members

Under `DENDRO_MESH_REPLAY`, init WAMR is skipped (`BSSNCtxGPU` initialize loop), so members with different ID keep the coarse start mesh and only remesh at recorded flag steps.

Validated on c101-014: perturbed `BSSN_BH1.MASS=0.4830` under replay of fiducial mesh — no init remesh, frozen remesh at steps 2 and 4, `C_HAM` shifts vs fiducial as expected.

## 2026-11-09 correction: init-grid converge was NOT replayed (bug)

Replay used to skip init WAMR entirely, so a replayed run started on the *unrefined* start mesh while the
recorded fiducial started on the converged init mesh. Recorded per-step flags were then applied to the
wrong octree. This was invisible on the coarse q1 smoke (init refinement trivial) but broke the
fine-resolution case: fine fiducial *record* C_HAM = 1.008e-5 vs *replay of the same mass* 5.04e-6
(`tier15_dualres_c104-019.log`). **The Tier 1.5 dual-res "FD failure" (37x disagreement) is therefore a replay
artifact, not evidence against FD.**
Fix: init iterations are recorded as `mesh_flags_-<iter>_*` and replayed (`record_init_flags` /
`apply_init_flags`; MPI_Abort on size mismatch). Old record dirs without init flags behave as before.
Re-test: `run_tier15_dualres_v2.sh` (includes record-vs-replay self-check). Built OK, NOT yet run (no GPU node).
