# Tier 3.1 — Tangent-linear RHS (plan)

## Why now

Tier 1.5 FD dual-res: coarse ∂C_HAM/∂θ ≈ −1.66e-6 vs fine ≈ −6.15e-5 — **no agreement**. Short C_HAM plateau alone is not enough for inversion Jacobians. TL (or adjoint) is required for trustworthy derivatives.

## Goal

Differentiated BSSN RHS: given background state `u` and K tangent vectors `v[k]`, compute `Jv = (∂RHS/∂u) v` sharing stencil work with the fiducial RHS where possible. Verify vs FD on the 1.5 plateau region (single-res, small θ).

## Codegen entry points

| piece | path |
|-------|------|
| SymPy BSSN | `CodeGen/bssn.py`, `bssn_eqns_*.inc.cpp` |
| GPU point algebra | `BSSN_GR/cuda/scripts/gen_gauge6_point.py` → fused PT kernels |
| Ensemble K (v1) | `grid.y=K` already batches members — TL can ride as extra K slots or separate tangent batch |
| Frozen mesh | Tier 1.1 replay — required so FD/TL see identical geometry |

## Work packages

1. **Symbolics** — differentiate RHS expr w.r.t. state + 1st/2nd derivs (product rule on metric/Christoffel/Ricci). Emit C++/CUDA point kernel `bssnrhs_tl` taking `(u, Du, DDu, v, Dv, DDv)`.
2. **Derivatives of tangents** — reuse existing FD stencil on `v` (same unzip path); no new unzip for geometry.
3. **Driver** — `BSSNCtxGPU` mode: fiducial + K tangents, one remesh history (1.1), deterministic TPID (1.2).
4. **Verify** — vs FD at plateau ε on q1 frozen mesh; report max |TL−FD|/|FD|.
5. **Gauge modes** — expect growth near punctures; optional tangent renormalization.

## Effort / risk

- Large codegen (days), but FD gate already failed — this is on the critical path for inversion.
- Ensemble K infrastructure (1.3) is a force multiplier once TL kernel exists.

## Status

- Scaffold: `BSSN_GR/cuda/include/bssn_tl_stub.cuh` (not linked into solver).
- **Harness v0 PASS:** `tl_gauge_fd_test` — host stencil, rel ~1e-8–1e-5.
- **Harness v1 PASS:** `tl_gauge_device_test` — device `__pt_fused_*3` stencils, rel ~1e-8 on c125-022 (`tier0/next_gpu_task.log`).
- **Harness v2 PASS:** `tl_gauge_q1_test` on real frozen-mesh q1 dump (`DENDRO_DUMP_TL_GAUGE`, blk=5 interior, nx=13, dx≈16.7). eps 1e-6/1e-7 GATE PASS (`tier0/tier31_q1_tl.log`).
- **Harness v3 PASS:** `tl_gauge_radbc_test` — LEFT-face radiative BC TL vs FD (face + interior), eps 1e-6/1e-7 on c106 (`tier0/tier31_radbc.log`). Helper: `device::tl::radiative_bc_tl` in `bssn_tl_stub.cuh`.
- **Harness v4 PASS:** `tl_shift_beta_test` — interior β⁰ TL (product rule on β·∇β⁰ + KO), eps 1e-6/1e-7 on c106 (`tier0/tier31_shift_beta.log`).
- **Harness v5 PASS:** `tl_Bdriver_test` — Γ-driver B⁰ TL: δB⁰ (damping+advection+KO) and δβ⁰ coupling, eps 1e-6/1e-7 on c106 (`tier0/tier31_Bdriver.log`).
- **Codegen scaffold PASS:** `CodeGen/bssn_tl_gauge.py` — symbolic TL of α/β⁰/B⁰/χ matches harness closed forms (7/7). Emit: `bssn_tl_gauge.inc.cuh`.
- **Harness v6 PASS:** `tl_chi_test` — χ TL vs FD (δχ, δα, δK), eps 1e-6/1e-7 on c106 (`tier0/tier31_chi.log`).
- **Codegen 10/10 PASS** incl. gt00 (δα, δAt00, weighted-Lie δgt00).
- **Harness v7 PASS:** `tl_gt_test` — gt00 full Lie TL vs FD (δgt0, δgt1 cross, δα, δAt0), eps 1e-6/1e-7 on c106 (`tier0/tier31_gt.log`). SymPy 11/11.
- **Tier 3.2 PASS:** maxdepth Richardson q4 — see `TIER32_RICHARDSON.md`.

### Next coding slice

1. ~~gauge / χ / gt00 full Lie~~ **done**.
2. At/K/Gt TL codegen + device gates; then fused emit.
3. Tier 3.3 hybridization-length (orbits) knob.