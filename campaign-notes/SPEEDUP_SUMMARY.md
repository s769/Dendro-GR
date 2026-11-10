# Speedup vs baseline — 2026-11-09 (GB200, FP64, all MEASURED unless marked "derived")
Baseline = stock fused kernel (`eval_rhs3`, no env knobs), original extraction path, host-staged ghosts (pre-campaign).

## Kernel / step level
| item | baseline | now | speedup | where |
|------|---------:|----:|--------:|-------|
| RHS microbench `rhsgpu_test 2048 20 0` (1 GPU) | 0.107-0.116 s | 0.044-0.050 s (PT_ASYNC, lb(343,1)) | **~2.3x** | `GB200_RESULTS.md` |
| q8 meshprobe step, 4 GPU, T=0.5 (`run_speedup_ab.sh`): stock | 2.29 s | PT_ASYNC 1.90 s | 1.20x | `tier0/` |
| ... + boundary-kernel overlap (new, default on) | 1.90 s | 1.70 s | 1.11x (cum. **1.35x**) | `OPTIMIZATION_LOG.md` |
| CUDA-aware MPI ghosts (earlier agent; compile flag, default build) | 3.03 s | 2.13 s | 1.42x | `TIER2_SCALE.md` |
| **cumulative stepping vs original (stock + host-staged)** | ~3.2 s (derived: 2.29 + 0.9) | 1.70 s | **~1.9x (derived)** | |
| while-hot kernel (opt-in) | 1.90 | 1.89 | none end-to-end (microbench -28%) | |
Note: ETS in these short runs includes ~0.25 s one-time startup; steady-state ratios are slightly larger (~1.4x for the measured part).

## Extraction (waveform/constraints)
| case | baseline | now | speedup |
|------|---------:|----:|--------:|
| q8 meshprobe 4 GPU, extraction every 4 steps (30 extractions) | 48.4 s | 13.7 s | **3.5x** (extraction itself 35.8 -> 3.2 s, 11x) |
| coarse q1 1 GPU, 15 extractions | 33.8 s | 13.05 s | 2.6x |
| production-depth q8 4 GPU, 1,729 steps, extraction every 100 | 159.8 s | 132.0 s | 1.21x (extraction -95%) |
psi4 agreement with original: 8e-11..3e-9 rel (print precision); shell path alone 1e-17 abs. Needs `DENDRO_GW_SHELL_ONLY=1` + OMP threads (cache is default on).

## Per-simulation / ensemble
| item | result |
|------|--------|
| production-depth q8 pure stepping (18.7k elements, dt 1e-3) | 0.044 s/step |
| ensemble member under replay (no remesh test, ckpt off, shell extraction) | ~0.045 vs 0.092 s/step => ~2.0x (derived from components, not an end-to-end run) |
| ensemble RHS fused launch (`grid.y=K`) | K=8: 2.9x vs serial at 32 blocks; 1.7x at 64; ~1.0-1.17x at >=256 |
| frozen-mesh replay fix | removes a replay bug (fine-res init mesh); fiducial/replay C_HAM agree 1e-12 |

## What did NOT help (details in notes)
LTS on GPU (no-go; ceiling 1.7-1.95x under default LTS offset), unzip hang/nh split & coalescing variants, async-K, defer-DD / scoped / CSE staging, K-members-per-CTA (v2), packed K ghost exchange (0.5-0.9x at these sizes),
UCX transport knobs, unzip/exchange overlap, FP32 variants (parked), tensor cores (closed).

## Adaptive remeshing (GPU BH_WAMR; opt-in `DENDRO_GPU_BH_WAMR=1 DENDRO_GPU_WAMR=1`)
| case | before | now | speedup |
|------|-------:|----:|--------:|
| remesh *test* on meshprobe (12 tests, 4 GPU) | 26.3 s (host unzip + host WAMR) | 0.47 s (device matrix-free WAMR) | 56x |
| Mesh construction per remesh, meshprobe late mesh (max rank) | 2.37 s | 0.37 s | 6.4x |
| meshprobe adaptive stress (remesh every 10 steps, 12 remeshes) | 82.6 s | 27.8 s | 3.0x |
| production-depth q8 adaptive (35 remeshes, 1,729 steps) | 380.8 s | 190.4 s | 2.0x (7.5 s -> 1.4 s per remesh) |
Adaptive vs non-adaptive production-depth run: 190.4 s vs 132.0 s (+44%); before the remesh work it was +189%.

## Per-simulation fixed cost (production-depth q8, 4 GPU)
| item | before | now |
|------|-------:|----:|
| `init_grid` initial-data evaluation (host, per simulation / ensemble member) | ~584 s | **74 s** (OpenMP, 8 threads/rank; 7.9x) |
| time from launch to first step | ~700 s | **~108 s** (6.5x) |
Var-batched unzip (`DENDRO_UNZIP_VB`): no-go (+5..13% slower). K-member batching of RHS/unzip/ghost: closed (see `BATCHING.md`).

## Biggest caveat
GPU runs with production parameter files (`BSSN_REFINEMENT_MODE = 4`) never remesh (GPU `is_remesh` lacks BH_WAMR). With the gated fix (`DENDRO_GPU_BH_WAMR=1`) the first version ran ~10x slower on the meshprobe; after the device WAMR test and the block-setup fix the adaptive overhead is +44% at production depth (see above).
The default is still OFF pending your decision; remeshing correctness over long runs has only been checked for short windows.

## CORRECTED headline measurement (2026-11-10) -- production depth q8, T=0.3 (~300 steps, 4 GB200 GPUs, same code revision)
(run_speedup3.sh; "ETS" = stepping phase incl. remeshes and 3 GW extractions; startup = TwoPunctures read + init_grid + init refinement)
| configuration | startup | ETS | remeshes | per-step (ETS minus remeshes)/300 | 300-step wall |
|---|---|---|---|---|---|
| ORIGINAL port: stock RHS, serial init, host unzip vectors, no overlap, **no remeshing** (BH_WAMR not implemented), original physics | 693 s | 30.6 s | 0 | 102 ms | ~724 s |
| optimized, ORIGINAL physics (pure performance work) | 43 s | 29.0 s | 5 (mean 1.35 s) | 74 ms | ~72 s |
| optimized, CORRECTED physics (init-mesh fix + SSL + CAHD + per-step BH tracking) = production | 57 s | 35.1 s | 5 (mean 1.10 s) | 99 ms | ~92 s |
Reading:
* Startup: 693 s -> 43 s (16x) from OpenMP init + lazy vectors etc.; with corrected physics 57 s (larger init mesh).
* Per-step (compute) cost: optimized/original-physics is 1.4x faster than the original port (74 vs 102 ms, and that includes the
  remeshing machinery the original never ran). With the CORRECTED physics the per-step cost (99 ms) is ~equal to the original port's:
  CAHD (+12%), the larger init mesh, BH tracking (+5%) and SSL use up the per-step gain.
* The original port never remeshed under the production parameters (BH_WAMR unimplemented), i.e. it ran a static mesh: it is NOT a
  scientifically usable baseline; its per-step number is flattering for it.
* For a 300-step run the end-to-end gain is 724 s -> 92 s (7.9x) dominated by startup; for long runs the per-step ratio dominates
  (~1.0x at equal physics vs the original port, 1.4x at original physics).  GPU vs CPU: smoke case to t=14, 165 s vs 1262 s (36 CPU ranks).
Caveats: single samples, only 300 steps; per-step includes 3 extractions (shell-only in the optimized runs, full host extraction in the original).
