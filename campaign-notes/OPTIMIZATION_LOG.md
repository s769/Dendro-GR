# Full-step optimization log (2026-11-09 onward)

Goal: speed up the whole step (not only RHS) in the regime the inverse problem needs; every change env-selectable; report bit-id.

## Diagnosis (q8 meshprobe, 4 GPU, CUDA-aware, PT_ASYNC; nsys `nsys_bdy1`, dev0)
GPU busy 64-71% of the 1.84 s step. Kernel time (s/rank): PT RHS 0.55-0.61, unzip 0.41-0.51, stock-boundary RHS 0.36-0.39, ghost pack/unpack 0.14-0.18.
Idle gaps total 0.6 s/rank: **0.42 s = `__write_sb_cg__` -> `__read_rb_cg__` (ghost exchange, 845 us per stage)**; 0.08 s enforce_evar->enforce_evar (122 gaps, 634 us); rest small.
The boundary kernel (37-111 CTAs < 148 SMs) is latency-bound: ~0.63 ms/launch even though it holds 3-10% of blocks.

## Results
| # | change | knob | q8 4GPU ETS (s) | bit-id | verdict |
|---|--------|------|----------------|--------|---------|
| 1 | launch stock-boundary kernel FIRST on a side stream so it overlaps the interior PT kernel (disjoint blocks) | `DENDRO_BDY_STREAM` (**default ON**, `=0` off) | 1.92 -> 1.70 (**-11%**) x2 repeats | GW psi4 output byte-identical A/B | **KEEP, default on** |
| 2 | unzip ghost-independent elements while exchange in flight (`unzip_cg_part`, indep/dep lists in MeshGPU) | `DENDRO_UNZIP_OVERLAP=1` (default off) | 1.70 -> 1.70 (A 1.701/1.739, B 1.699/1.719) | per-element same kernel | **NO-GO** (only 54-68% of elements are ghost-independent at these partition sizes; exchange wait not hidden) |
| 3 | UCX settings: TLS=cuda_ipc,cuda_copy,sm,self ; RNDV_SCHEME=put_zcopy ; RNDV_THRESH=inf | env | 1.74->1.79 ; 1.71->1.88 ; 1.73->9.30 | n/a | **NO-GO** (default best) |

Noise note: Constraints (and ~1e-18 abs in psi4) differ run-to-run even for identical serial runs; unrelated to these changes (A vs A2 vs A3 all differ).
Data: `tier0/ab_bdystream.log`, `tier0/ab_unzipov.log`, `tier0/ab_ucx*.log`.

## Extraction-path cost (found while explaining slow waveform replays)
Waveform/constraint extraction (`BSSNCtxGPU::write_vtu`) runs on the HOST: CPU unzip of all 24 vars, single-core `physical_constraints` over every block (shared global
scratch `BSSN_DERIV_WORKSPACE` => not thread-safe), zip, ghost, psi4 interpolation. Timers (`DENDRO_TIME_EXTRACT=1`), coarse q1, 3.4k elements, 15 extractions in T=4:
total 23.4 s of ETS 34.5 s (**68%**): unzip+D2H 8.4 s (0.56 s/call), constraints loop 11.1 s (0.74 s/call), zip+ghost 0.4, extractConstraints 1.0, psi4 2.5.
Cost grows with mesh size (replays at 40k elements ran ~10 s/step with extraction every 4 steps). Production pars extract every 50-100 steps (q4: 50, q8: 100), so at q8's ~4M steps
extraction is a plausible 10-30% of wall time — NOT measured at production scale; needs one profiled q8 run. Cheapest lever: `BSSN_GW_EXTRACT_FREQ` (parameter).

## Extraction fix (resolved 2026-11-09)
The earlier ~1% psi4 "discrepancy" of the shell-only path was a STATE LAG, not an unzip difference: the driver snapshots the host state with an async D2H at the
extraction step but runs `write_vtu` a step or more later, while my shell path unzipped the *current* device state. Fix: in shell mode the driver extracts INLINE at the
extraction step (`device_to_host_sync` + `write_vtu`), so state and time agree.

Four changes (all validated against the original path on identical inputs):
| # | change | knob | note |
|---|--------|------|------|
| 1 | psi4/constraints only on blocks intersecting the GW spheres (margin 2 block widths), device unzip + per-block D2H instead of host unzip of everything; skip constraint norms | `DENDRO_GW_SHELL_ONLY=1` (default OFF; constraint norms/`Constraints.dat` invalid in this mode) | `DENDRO_GW_SHELL_MARGIN` |
| 2 | `physical_constraints` thread-safe (thread_local scratch) + `physical_constraints_blocks` OpenMP loop | `OMP_NUM_THREADS` (run with `mpirun --bind-to none` / Slurm `--cpus-per-task`; bound-to-1-core made it SLOWER) | affects CPU solver path too, numerically identical |
| 3 | cached psi4 sphere sampler: octree search + 1D Lagrange weights cached per mesh, points grouped by element, thread-safe nodal gather | `DENDRO_GW_CACHE_SAMPLER` (default ON, `=0` original) ; `DENDRO_GW_CACHE_VERIFY=1` cross-checks vs original | signature must include `getElementLocalBegin/End` + total element count (a neighbour remesh shifts element ids) |
| 4 | shared-scratch bug avoided: default `getElementNodalValues` uses `RefElement::im_vec1/2` (not thread-safe) -> use the 6-arg overload with per-thread buffers | - | |

Results (extraction every 4 steps, `DENDRO_TIME_EXTRACT=1`):
| case | baseline ETS (s) | new ETS (s) | extraction (s) base -> new | psi4 agreement |
|------|-----------------:|------------:|---------------------------:|----------------|
| coarse q1, 1 GPU, 15 extractions | 33.8 | 13.05 (OMP=8, shell, cache) = **-61%** | 23.1 -> 4.2 (5.5x) | 3e-11..1.5e-10 rel (print precision) |
| **q8 meshprobe, 4 GPU, 30 extractions** | **48.4** | **13.7** = **-72%** | 35.8 -> 3.2 (**11x**) | 8e-11 (l2m2), 3e-10 (l2m1), 3e-9 (l3m2, tiny mode) rel; shell path alone matches to 1e-17 abs |

Production relevance (extraction every 50-100 steps): baseline cost ~1.2 s/extraction at q8 mesh size vs ~0.1 s/step => ~12% overhead at freq 100, ~1% after; larger meshes scale worse for baseline.
Recommended inversion-ensemble settings: `DENDRO_GW_SHELL_ONLY=1`, `OMP_NUM_THREADS=8..16` per rank with cores unbound, cache on (default). Use the default (full) path when constraint norms are needed.
Data: `tier0/*` logs from `run_extract_ab3.sh`, `run_extract_q8.sh`, `run_cache_verify.sh` (744 cache-vs-original checks, 0 mismatches after the signature fix).

## Production-depth q8 profile (2026-11-09) — MEASURED
Setup: `pars/q8.par.toml` (maxdepth 17, remesh test /50, GW extract /100, checkpoint /100) with TPID grid set to the reusable 60^3 solution and the same bare masses; 4 GPUs (one GB200 node),
CUDA-aware MPI, PT_ASYNC + boundary overlap. Init mesh 18,656 elements (4x the meshprobe), initial lmax 11 (dt 1.6e-2) refining to dt 1.0e-3 by step 0; run to t=1.8 = 1,729 steps.
Initial-mesh convergence (10 WAMR iterations + TPID load) takes ~9 min wall before step 0 (per simulation!). Script: `run_prod_q8.sh`; logs `tier0/prod_*.out`; analysis `analyze_prod_steps.py`.

| run | cadences on | ETS (s) | derived cost |
|-----|-------------|--------:|-------------|
| D | none (pure stepping) | 75.9 | **0.044 s/step** (47% of A) |
| F | checkpoint only | 81.1 | checkpoints +5.3 s (18 writes, 0.3 s each; 3%) |
| E | remesh test only | 117.6 | **remesh *test* +41.7 s (34 tests, 1.2 s each; 26% of A)** — no remesh actually triggered in this window |
| C | remesh test + ckpt | 130.7 | |
| A | all, original extraction | 159.8 | extraction +29 s (timer: 35.5 s; 15 calls, 2.4 s each; 18-22%) |
| B | all, **new extraction** (`DENDRO_GW_SHELL_ONLY=1`, cache, OMP=8) | 132.0 | extraction ~1.3 s (**-95%**); total -17% vs A |

Takeaways:
1. At production depth the host-side remesh *test* (`is_remesh()`: D2H of all state + host unzip + host WAMR every 50 steps) costs as much as 40% of stepping — the next biggest non-stepping cost. It is skipped in frozen-mesh replay mode, so ensemble members avoid it; the recording fiducial and non-frozen production runs pay it. A device unzip + device wavelet/refinement-flag kernel (same structure as the extraction fix) is the natural next target.
2. Extraction fix removes ~18% of the run; with replay (no remesh test), checkpoints off, and shell extraction an ensemble member runs at ~0.044 + ~0.001 s/step vs 0.092 s/step for production defaults here => **~2.1x per-member** in this window.
3. Checkpoints are cheap (3%); keep production cadence but turn them off for ensemble members.
4. Earlier meshprobe profiles (no remesh/extraction/ckpt, 4x smaller mesh, depth 15) describe pure stepping only.
5. ~9 min of per-simulation initial-mesh convergence precedes step 0; the frozen-mesh replay of init refinement (Tier 1.1 fix) lets perturbed members skip it. Not yet measured end-to-end.
Caveat: window is the first 1,729 steps of a ~4M-step run; AMR refines further later (more elements, remesh events actually trigger and cost extra), so shares will shift.

## !! GPU runs with the production parameter files never remesh (found 2026-11-09)
`BSSN_REFINEMENT_MODE = 4` (= `BH_WAMR`) in `pars/q1,q2,q4,q8,q12,q16,q24.par.toml` (q32 uses mode 3 = BH_LOC). The CPU `BSSNCtx::is_remesh` implements BH_WAMR
(`isRemeshBH` + `addRemeshWAMR`); `BSSNCtxGPU::is_remesh` has only WAMR, EH, EH_WAMR, BH_LOC — for mode 4 it falls through and returns "no refinement". Consequences:
- GPU production-parameter runs keep the INITIAL mesh for the whole run (refined around the initial puncture positions; does not follow the BHs). Not caught because no remesh events appeared in any
  profile (prod_A..F: 0 remeshes). The "remesh test" cost seen in the production profile (41.7 s / 26%) was only the host unzip of all vars before falling through.
- Whether this was intentional for earlier GPU production runs is unknown (LS6 logs predate this tree?). Needs your decision.
Added (default OFF = historical behaviour): `DENDRO_GPU_BH_WAMR=1` mirrors the CPU branch in `BSSNCtxGPU::is_remesh`.
Measured on q8 meshprobe (depth 15, 4 GPU, remesh test every 10 steps, mode 4, T=0.5):
| gate | remeshes | ETS (s) | breakdown |
|------|---------:|--------:|-----------|
| off (today) | 0 | 5.5 | host unzip 3.3 s wasted |
| on (`DENDRO_GPU_BH_WAMR=1`) | 12/12 | 59.0 | test 26.7 s (host unzip 8.7 + BH/WAMR 18.0) + 12 actual remesh/grid-transfer ~27 s (~2.2 s each) |
=> correct adaptive behaviour costs ~10x the current GPU behaviour at this cadence, all host-side (host unzip, single-thread WAMR, remesh/grid transfer). This, not RHS, is the real production lever once BH_WAMR is on.

## Device WAMR remesh test (matrix-free) — 2026-11-09
`DENDRO_GPU_WAMR=1` (with `DENDRO_GPU_BH_WAMR=1`, mode BH_WAMR): the wavelet refinement test runs on the GPU from the device-unzipped state, no host unzip and no host `compute_wavelets_3D`.
Kernel `wamr_lmax_kernel` (one CTA per element): interior octants: 7^3 coarse (every 2nd of the 13^3 block) -> degree-6 Lagrange prediction at the 6^3 odd points,
wc=|u-pred|/max(1,|u|); boundary octants: 4^3 coarse -> degree-3 prediction at 3^3 points (27 coeffs, normalised by sqrt(216) as the host does); metric = sqrt(sum wc^2)/sqrt(216), max over the 24 refine vars.
Host side `bssn::addRemeshWAMR_lmax` applies exactly the `addRemeshWAMR` flag logic (1e12 tol amplification inside BH AMR radii etc.) from the precomputed metrics.
Validation (`DENDRO_GPU_WAMR_VERIFY=1`, host and device on identical input flags): **0 flag mismatches** over 12 tests x 4 ranks (host SPLIT counts 5..220/rank); psi4 of whole run agrees to 1e-18 abs.
Pitfall found: skipping the host unzip also skipped the host ghost exchange of `m_evar`, which the host grid transfer needs (device RK never writes ghosts) -> NaN after the 2nd remesh; fixed with an explicit `readFromGhost` on the skip path.
Also removed a redundant second device-mesh upload in `remesh_and_gridtransfer` (grid_transfer already rebuilt it for the new mesh; `DENDRO_REMESH_DOUBLE_UPLOAD=1` restores): psi4 identical (2e-19); device mesh alloc is only ~4 ms here, so negligible gain.

Results, q8 meshprobe depth 15, 4 GPU, mode 4, remesh test every 10 steps (12 tests, 12 actual remeshes), psi4 every 10:
| config | remesh test total | ETS (s) |
|--------|------------------:|--------:|
| host BH_WAMR (R2) | 26.3 s (host unzip 8.4 + host WAMR 17.9) | 82.6 |
| device WAMR (R3) | **0.47 s** (56x) | **55.4** (-33%) |
Remaining remesh cost: per event ReMesh 0.56-0.77 s + grid transfer ~0.5 s (max over ranks ~1.1-1.5 s) -> ~16 s of the 55.4 s; all host (octree ReMesh + new Mesh construction + host grid transfer).
Recommended for adaptive production runs: `DENDRO_GPU_BH_WAMR=1 DENDRO_GPU_WAMR=1`.

## Mesh construction during remesh: O(N^2) block setup fixed (2026-11-09)
`Mesh::performBlocksSetup` (dendrolib, called from `new Mesh` in `ReMesh`) ran, for EVERY block, `SFC_treeSearch(blkKeys ~26 keys, ALL elements)`, which buckets the whole
element range => O(N) per block, O(N^2) overall. Stack sampling: 22/60 samples there; mesh ctor was the whole ReMesh cost (balance+partition only ms) and grew superlinearly with mesh size (0.18 s @4.6k elem -> 3.6-6.6 s @ ~50k elem prod-depth, 30% rank imbalance).
Fixes (dendrolib `src/mesh.cpp`, in `patches/dendrolib/current_local_full.patch`):
1. loop over blocks parallelised with OpenMP (per-block data only); `DENDRO_BLKSETUP_SERIAL=1` restores serial.
2. key->element lookup by binary search on the SFC-sorted element array (leaf containing a max-depth key = last element <= key, `isAncestor` check): O(log N); `DENDRO_BLKSETUP_TREESEARCH=1` restores `SFC_treeSearch`.
   `DENDRO_BLKSETUP_VERIFY=1` runs both and compares: **0 mismatching key results** over all blocks/ranks/12 remeshes.
Results (q8 meshprobe depth 15, 4 GPU, mode 4 + device WAMR, remesh test every 10 steps, 12 remeshes, psi4 every 10):
| config | Mesh ctor (late remesh, max rank) | ReMesh | ETS (s) |
|--------|---------------------------------:|-------:|--------:|
| serial, tree search | 2.37 s (min 1.68) | 2.52 s | 55.4 |
| OMP=8, tree search | 1.03 s | 1.01 s | 33.9 |
| OMP=8, binary search (new default) | **0.37 s** (min 0.36) | 0.40 s | **27.8** (-50% vs serial) |
psi4 agrees with the serial/tree-search run to 4e-19..1e-18 abs. Cumulative for the adaptive stress case: host BH_WAMR 82.6 s -> 27.8 s (3.0x).

## Production-depth q8, adaptive (BH_WAMR ON), 4 GPU, 1,729 steps, 35 remeshes (2026-11-09)
| run | stack | ETS (s) | per-remesh cost |
|-----|-------|--------:|----------------:|
| B (non-adaptive, as GPU runs today) | PT_ASYNC, new extraction | 132.0 | - |
| G | + BH_WAMR ON, device WAMR test, tree-search block setup (serial) | 380.8 | 7.5 s (ReMesh 3.6 + rest) |
| **G2** | + OpenMP/binary-search block setup (`Mesh ctor` 0.31 s) | **190.4** | **1.4 s** (ReMesh 0.6-0.7, grid transfer ~0.75) |
=> adaptive remeshing (35 events) now costs +58 s (+44%) over the non-adaptive run instead of +249 s (+189%); a remesh is ~1.4 s ~ 32 steps of stepping (was ~170).
Remesh *test* (device WAMR) total over 33 tests: 2.3 s (host version ~2 s per test at this size). Remaining per-remesh cost: host grid transfer (~0.45 s DVec::grid_transfer + ~0.3 s vector/mpi-ctx re-creation) and rest of ReMesh (~0.3 s).
Run H (host WAMR at production depth) crashed: I rebuilt the binary while it was starting (bus error), not a code issue; meshprobe host-vs-device comparison stands (26.3 s -> 0.47 s test).
Env summary for adaptive production runs: `DENDRO_GPU_BH_WAMR=1 DENDRO_GPU_WAMR=1`, OMP threads >1 (`--bind-to none`/`--cpus-per-task`); defaults changed in dendrolib: binary-search block setup ON (verify/restore knobs above), redundant 2nd device-mesh upload OFF (`DENDRO_REMESH_DOUBLE_UPLOAD=1` restores).

## DEFAULT CHANGE (2026-11-09, your decision): GPU BH_WAMR remeshing + device WAMR are now ON by default
`BSSN_REFINEMENT_MODE=4` runs on the GPU now remesh like the CPU driver. Opt out: `DENDRO_GPU_BH_WAMR=0` (old no-remesh behaviour), `DENDRO_GPU_WAMR=0` (host WAMR test). Verified default-on: meshprobe mode 4, 12/12 remeshes, ETS 28.0 s.
Still to do: a longer-run physics comparison of GPU adaptive vs CPU path (only short windows checked so far).
