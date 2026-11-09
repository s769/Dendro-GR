# Tier 1.2 — Deterministic TwoPunctures / inversion parameter vector

## Differentiated vector (ensemble / SVN)

With `TPID_GIVE_BARE_MASS = 1` (required for the inversion path):

| Symbol | Source | Notes |
|--------|--------|-------|
| \(m_\pm\) | `BSSN_BH{1,2}.MASS` | Bare puncture masses → `TPID::par_m_{plus,minus}` |
| \(\mathbf{P}_\pm\) | `BSSN_BH{1,2}.V_{X,Y,Z}` | Linear momenta |
| \(\mathbf{S}_\pm\) | `BSSN_BH{1,2}.SPIN` + angles | Cartesian spin from \((S,\theta,\phi)\) |
| \(b\) | `TPID_PAR_B` | Half-separation; punctures at `center_offset ± (b,0,0)` |
| \(\mathbf{c}\) | `TPID_CENTER_OFFSET` | Puncture midpoint |

Frozen / out of the differentiated path:

- `Newton_maxit = 15` (compile-time), `TPID_NEWTON_TOL` (par, keep fixed across ensemble)
- Spectral grid: `TPID_NPOINTS_{A,B,PHI}`
- `TPID_SOLVE_MOMENTUM_CONSTRAINT` (keep 0 unless intentionally on)
- Eccentricity reduction: **not in solver**; bake Healy/PN quasi-circular \((P,S)\) into the par offline
- ADM→bare mass iteration: **disabled** when `GIVE_BARE_MASS=1`. Do not set it to 0 for ensemble members

## Geometry vs tracking

TwoPunctures places punctures at `center_offset ± par_b`. `BSSN_BH{1,2}.{X,Y,Z}` drive AMR / BH tracking and need not equal the puncture coords (RIT q1: BH at \(3/-4\), \(b=4\)). Rank-0 logs a NOTE when they differ by >1M.

## Code

`bssn::resolveTPIDBareParameters` (end of `readParamFile`) wires bare masses from BH MASS when `GIVE_BARE_MASS=1` and prints the resolved vector. Previously `par_m_*` always came from `TPID_TARGET_M_*`, so q4/q8 meshprobe pars silently used equal-mass TPID.

## Validation (c124-009)

| Case | Result |
|------|--------|
| q1 dual `tpid` | **BIT_IDENTICAL** `md5=3ef70d0a4538a4afcbb989afaeba3cb2` (matches prior `rit_q1` sol) |
| q4 dual `tpid` | **BIT_IDENTICAL** `md5=e095a2af3b713939d99dcc4a78ea9bcb`; bare `m+=0.2 m-=0.8` |

Artifacts: `Dendro-GR-runs/tpid_tier12/`, log `tpid_tier12_run.log`.
