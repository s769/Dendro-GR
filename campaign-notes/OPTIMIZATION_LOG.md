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
