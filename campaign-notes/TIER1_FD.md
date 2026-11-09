# Tier 1.5 — FD Jacobian scout (q=1 frozen mesh)

## Setup (c101-014)

- Frozen mesh replay from fiducial `tier14/mesh_rec`
- Frozen `rit_q1_tpid_sol.bin` (bare-mass eval only; full TP re-solve not in this scout)
- Observable: final `C_HAM` L2 after 5 ETS steps (`q1.smoke_cons`)
- Parameter: `BSSN_BH1.MASS` about \(m_0=0.4824\)

## Central / forward FD

| ε | central \(d C_{HAM}/dm_+\) | forward |
|---|---------------------------:|--------:|
| 1e-4 | **-1.159403e-6** | -1.159300e-6 |
| 3e-4 | **-1.159403e-6** | -1.159095e-6 |
| 1e-3 | **-1.159404e-6** | -1.158379e-6 |
| 3e-3 | -1.159412e-6 | -1.156337e-6 |
| 1e-2 | -1.159509e-6 | -1.149255e-6 |

**Plateau:** central FD stable to ~1e-12 relative across ε ∈ [1e-4, 1e-3]. Forward bias grows with ε as expected.

## Gate implication (C_HAM)

For this short-run constraint observable, **FD is sufficient** (clear plateau).

## Waveform scout (c104-019, 1 GPU, `GW_L2` |ψ| at r=50)

| ε | central | forward |
|---|--------:|--------:|
| 1e-4 | -1.409e-6 | -1.409e-6 |
| 3e-4 | -1.174e-6 | -1.194e-6 |
| 1e-3 | -1.198e-6 | -1.204e-6 |
| 3e-3 | -1.209e-6 | -1.221e-6 |
| 1e-2 | -1.198e-6 | -1.198e-6 |

Plateau is weaker than C_HAM (ε=1e-4 outlier; ~[-1.17,-1.21]×10⁻⁶ for ε≥3e-4). Short 5-step smoke + frozen TP sol — need longer evolve / 2nd resolution before deciding Tier 3. Raw: `tier0/tier15_gw_fd_c104-019.log`.

## Longer evolve (TIME_END=1.0, 16 steps, c104-019)

Observable: `|ψ₄|` L2 at r=50 (`GW_L2_r50_mag`); also C_HAM for cross-check.

| ε | central d(GW)/dm₊ | central d(C_HAM)/dm₊ |
|---|------------------:|---------------------:|
| 3e-4 | -1.409e-6 | -1.379e-6 |
| 1e-3 | -1.268e-6 | -1.662e-6 |
| 3e-3 | -1.268e-6 | -1.474e-6 |

GW central FD agrees at ε=1e-3 and 3e-3 (−1.268e-6); ε=3e-4 still an outlier. C_HAM under longer evolve is **not** as flat as the 5-step scout (−1.159e-6 plateau) — longer window mixes gauge/constraint dynamics. Raw: `tier0/tier15_gw_fd_long_c104-019.log`.

## Dual-res FD (ε=1e-3, TIME_END=1.0, C_HAM)

Scripts: `run_tier15_dualres.sh` / `run_tier15_dualres_resume.sh`. Fine mesh = `q1.smoke_hfine` record; coarse = `tier14/mesh_rec`.

| res | C_HAM(m₀) | C_HAM(m₀+ε) | C_HAM(m₀−ε) | central FD |
|-----|----------:|------------:|------------:|-----------:|
| coarse | 5.863320e-6 | 5.861942e-6 | 5.865266e-6 | **−1.661732e-6** |
| fine | 5.042178e-6 | 4.057276e-6 | 4.180344e-6 | **−6.153405e-5** |

Coarse FD matches the longer-evolve C_HAM column at ε=1e-3 (−1.662e-6). Fine mesh (hfine record) does **not** agree (~37× larger |FD|) — dual-res gate fails for longer-evolve C_HAM; treat FD as scout-only until a better observable / longer consistent window. `fine_m` finished on c127-032 (job 86503); ham extract recovered from Constraints.dat. Logs: `tier0/tier15_dualres_c104-019.log`, `tier0/tier15_dualres_resume.log`.

Also on c104-018: ensemble K refresh (64bl K=8 → 1.55× serial-K; 256bl → 1.20×) + 4-GPU q1 smoke ETS≈3.98s; q4 short 4GPU ETS≈4.42s, GTS/LTS ceiling≈10.7× (`tier0/q4_short_4gpu_c104-018.log`).

## Tier 2.4 short profiles (4 GPU, c104-018)

| case | N_tot | ℓ span | GTS/LTS ceiling | ETS (s) | wall (s) |
|------|------:|--------|----------------:|--------:|---------:|
| q4 short | 4390 | 3–12 | 10.665 | 4.42 | 17 |
| q8 short | 4670 | 3–13 | 14.251 | 8.94 | 23 |

q8 hist bit-matches prior 1-GPU `mesh_q8.log`. Artifacts: `tier0/q4_short_4gpu_c104-018.log`, `tier0/q8_hist_4gpu.log`, `run_q8_hist_4gpu.sh`.

## 2026-11-09 REVISED dual-res FD (after init-mesh replay fix) — supersedes the "fails" verdict above

Cause of earlier 37x disagreement: replay skipped init-grid refinement (see `TIER1_MESH_REPLAY.md`).
Re-run (`run_tier15_dualres_v2.sh`, `run_tier15_dualres_scan.sh`; c110-001/c103-009; logs `tier0/dualres_v2_*.log`, `tier0/dualres_scan_*.log`).
Observable: C_HAM L2 at TIME_END=1.0, q1, frozen mesh, d/d(BH1.MASS) about 0.4824, central FD.

Self-check (record vs replay, same mass): rel diff **1.8e-13 (coarse), 1.0e-12 (fine)** — replay now faithful.

| eps | coarse dC_HAM/dm | fine dC_HAM/dm |
|----:|-----------------:|---------------:|
| 3e-4 | -4.210e-6 | -3.907e-6 |
| 1e-3 | -4.143e-6 | -3.514e-6 |
| 3e-3 | -4.082e-6 | -3.757e-6 |
| 1e-2 | -4.009e-6 | -3.862e-6 |

Coarse: smooth O(eps^2) drift, +-2.5% over 1.5 decades -> clean plateau. Fine: +-5%, non-monotone (eps=1e-3 low) — small unexplained non-smoothness, not yet diagnosed.
Coarse vs fine differ ~10% (discretization-level), same sign/magnitude. **Dual-res gate: PASS for C_HAM (loose, ~10%).**
Caveats: one parameter, one observable (C_HAM L2 — a constraint norm, not the waveform), 16/31 samples. Waveform (psi4 phase) dual-res FD NOT yet done — that is the observable that matters for inference.
**Decision (per handoff 6.4): do NOT build the full TLM now.** Existing TL gates stay as scaffolding. Re-open only if waveform-phase FD plateau (next) is poor.

## 2026-11-09 Waveform FD (psi4_22, coarse q1, frozen mesh) — first result
Setup: `run_tier15_waveform_fd.sh` (SKIP_FID replay), coarse `q1.smoke_cons` (maxdepth 11), GW radii [10,12,15,18,20,25], REMESH_TEST_FREQ=20,
TEND=26, central FD in BH1.MASS about 0.4824, eps in {1e-3, 3e-3}. Fiducial + mesh record from `tier15_wave_coarse_T30_r3` (the T=30 fiducial
CRASHED at the step-420 remesh: mesh explodes 3k -> 68k elements, then "invalid arguments"; likely wave-driven refinement near the GW shell, cause not proven;
so all replays stop at T=26 < step 420). Raw: `tier0/wave_rep.log`. Analysis: `analyze_wave_fd.py`, `analyze_wave_multi.py`.

Observable: complex dpsi4_22/dm over the full 0<=t<=26 window. eps-plateau = ||dz(1e-3)-dz(3e-3)|| / ||dz(1e-3)||:

| radius | plateau (rel L2) | note |
|-------:|-----------------:|------|
| 10 | 3.8e-3 | |
| 15 | 9.0e-3 | |
| 25 | 3.4e-4 | |

=> **complex psi4 derivative FD plateau is good (<1%) between eps 1e-3 and 3e-3 on this window.** Phase-only derivative is ill-conditioned where |psi4|->0 (e.g. r=15: -574 vs -194) — use the complex derivative / amplitude-weighted phase, not raw arg().
CAVEATS: (1) this "q1 smoke" case is coarse and the window is dominated by an unphysical ~100x burst (C_HAM rises 1e-5 -> 6e-4 by t~25; |dz|/|z| ~ 50 at late times), not inspiral phase;
(2) only 2 eps values here (eps 3e-4, 1e-2 replays on c106-031 were cut off by node expiry — partial data only, see below if appended); (3) coarse resolution only — fine-res waveform FD NOT done;
(4) run-to-run noise: psi4 output differs ~1e-18 abs between identical serial runs (norm/extraction noise, evolution otherwise reproducible to print precision) — far below the FD signal.
Decision stands: TLM not needed on this evidence; re-open if a well-resolved physical window shows a poor plateau.

### Update: all four step sizes (partial runs for 3e-4, 1e-2 cut at t=24.5 by node expiry; compared on matched windows)
Reference eps=3e-3, complex dpsi4_22/dm, relative L2 difference over the common window:

| radius | eps=3e-4 | eps=1e-3 | eps=1e-2 |
|-------:|---------:|---------:|---------:|
| r=10 | 4.2e-3 | 3.8e-3 | 4.2e-2 |
| r=15 | 7.4e-3 | 5.5e-3 | 7.2e-2 |
| r=25 | 1.8e-4 | 3.4e-4 | 1.7e-3 |

**FD plateau: flat (<1%) for eps in [3e-4, 3e-3]; eps=1e-2 shows 2-7% truncation bias.** Recommended inversion FD step for BH1.MASS about 0.48: eps ~1e-3 (central).
Still caveated: coarse resolution, unphysical burst-dominated window, single parameter. Fine-resolution waveform FD not done.
