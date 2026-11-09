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
