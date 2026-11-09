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

Plan only. Next implementable step: prototype TL for a **scalar advection** or **gauge-only** subset to validate the codegen harness, then scale to full BSSN.
