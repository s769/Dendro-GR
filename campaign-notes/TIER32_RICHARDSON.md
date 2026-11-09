# Tier 3.2 — Resolution as a clean knob (q=4 Richardson sample)

## Setup (c106-029, job 86880)

- Base: `q4.meshprobe.par.toml` + `tpid_tier12/q4_a/mesh_q4_a_tpid_sol.bin`
- Only knob: `BSSN_MAXDEPTH` / `BH*_MAX_LEV` ∈ {13, 14}
- `BSSN_RK_TIME_END = 0.27`, `DENDRO_MESH_RECORD` per depth, `GW_EXTRACT_FREQ=1` for Constraints
- Script: `run_tier32_richardson_q4.sh` · log: `tier0/tier32_richardson_q4.log`

## Grid (after init converge)

| depth | lmin–lmax | dx | ETS (s) |
|------:|-----------|---:|--------:|
| 13 | 3–11 | 0.065104 | 45.4 |
| 14 | 3–12 | 0.032552 | 158.1 |

Finest spacing halves with +1 maxdepth as expected.

## Observable: final C_HAM L2

| depth | C_HAM | n samples |
|------:|------:|----------:|
| 13 | 3.47879010e-06 | 17 |
| 14 | 3.46442039e-06 | 34 |

|fine−coarse| = **1.437e-08**. Richardson p=2 continuum ≈ **3.45963049e-06**.

## Gate

**PASS** for “resolution as a clean knob”: identical pars except maxdepth produce a coherent dual-res pair (lmax/dx/C_HAM). Discretization-error sample recorded for the hybrid model.
