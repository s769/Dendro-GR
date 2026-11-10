# GPU vs CPU physics check (2026-11-09) — PARTIAL, more runs queued

Case: RIT q1 smoke setup (`q1.smoke_cons.par.toml`, BH1.MASS=0.4824, frozen TPID sol), MAXDEPTH 13, `BSSN_REFINEMENT_MODE=4` (BH_WAMR), remesh test every 10 steps, extraction every 10, GW radii 10..25.
Runs (`run_physcheck.sh`): `cpu` = CPU `bssnSolver` 36 MPI ranks (adaptive, CPU BH_WAMR); `gpu` = GPU defaults (adaptive, device WAMR); `gpu0` = GPU with `DENDRO_GPU_BH_WAMR=0` (frozen init mesh).
GPU: adaptive 254.6 s with 122 remeshes (every test remeshes) vs frozen 95.9 s (T=20). CPU ran ~0.7 steps/s; node expired at t=6.35 so comparisons below are for t<=6.35.

| quantity (t<=6.35) | gpu vs cpu | gpu0 vs cpu | gpu vs gpu0 |
|---|---|---|---|
| BH position max abs diff | 5.7e-3 (bh1_y 0.5489 vs 0.5432 at t=6.2) | 5.7e-3 | ~0 until t~6 (separation differs 2.7e-2 by t=20) |
| BH separation max diff | 1.9e-3 (6.96809 vs 6.96795 final) | 1.9e-3 | |
| C_HAM final | 1.59e-5 vs 1.54e-5 | 1.59e-5 vs 1.21e-5 | at t=20: adaptive 2.1e-5 vs frozen 7.9e-6 |
| psi4 l2m2 max abs diff / max | 1.0e-6 / 3.3e-4 (0.3%) | same | 1.0e-2 of max by t=20 |
| psi4 l3m2 | 1.3e-6 / 8.6e-5 (1.5%) | 1.2e-6 | |
Observations:
- The adaptive and frozen GPU runs are indistinguishable from each other through t~6 and differ from the CPU run by ~1% in BH y-velocity (growing slowly: 1.1e-4 vs 4.7e-5 at t=0.16, 5.7e-3 at t=6.2). Remeshing logic is therefore NOT the source of the CPU/GPU gap at early times.
- BH position tracking differs systematically; cause not identified (candidates: CPU/GPU unzip hanging-node interpolation, BH-velocity extraction on lagged/ghost-refreshed host data, step-0 C_HAM on GPU = 1.006e-5 vs CPU 9.207e-6 suggests the GPU step-0 diagnostic uses pre-enforcement/pre-ghost state).
- At t=20 adaptive GPU has HIGHER constraint violation (2.1e-5) than frozen-mesh GPU (7.9e-6): remeshing every test (122 events) injects interpolation error. The CPU run did not reach t=20 so the CPU adaptive trend is unknown.
QUEUED (run automatically when a node starts: `watch_batch.sh` -> `run_next_batch.sh`): CPU run on a FROZEN mesh (identical mesh to gpu0, mode `cpu0`, T=6.4) to isolate pure solver-implementation differences; OpenMP thread sweep; TP warm start.
