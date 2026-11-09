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
