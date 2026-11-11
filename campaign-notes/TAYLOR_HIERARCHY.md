# Forward Taylor hierarchy for exact waveform parameter derivatives (B0 math note)

Status: design note, written before any implementation (2026-11-11). Nothing here is measured yet; cost numbers are *model estimates* and are labeled so.

## 1. Setting and decisions (recorded reasoning)

Semi-discrete system (after spatial discretization, on the *frozen fiducial mesh*): du/dt = F(u, Du, D²u), u(0) = u0(θ), observable h(t;θ) = O(u(t)) (psi4 modes on extraction shells, then strain/log-amplitude/phase). θ ∈ R^p, p < 10 (first set: q, χ1z, χ2z; the map to the bare TwoPunctures inputs must be documented separately).

**Decision: forward (tangent / Taylor-mode) differentiation, not reverse mode.**
- Outputs are *full time series* of several modes at several radii (O(10^3–10^4) numbers), so a reverse sweep per output component is unaffordable; reverse mode wins only for a scalar objective, and the inference needs the waveform and its derivatives, not a scalar.
- For J = ∂h/∂θ: forward costs p tangent members; reverse needs one adjoint per *output functional* (a scalar likelihood gradient costs one adjoint, but we need J itself for the Fisher metric and for the Taylor model).
- For the Fisher-metric derivative dG/dθ_k = (∂J/∂θ_k)ᵀC⁻¹J + JᵀC⁻¹(∂J/∂θ_k) we need *model second derivatives*; a second-order adjoint (forward-over-reverse) costs ≥ p reverse sweeps with checkpointing of the whole nonlinear trajectory (and the adjoint of remesh/transfer and of the gauge sector, whose stability is unknown). Second-order tangents have the same O(p²) member count but no checkpointing, are exact, and share one nonlinear trajectory.
- Finite differences are the *verification oracle*, not the production method: measured second-difference noise grows ~1000× late in the run (junk-window data; to be re-measured on the inspiral window).

Excluded by prior decision: ROM/POD of the tangent problem, FP32 state, tensor cores for the nonlinear RHS. (FP32 *storage of tangent members only* is a later, B3-gated experiment.)

## 2. The hierarchy

Write F = F(U) with U = (u, Du, D²u) collecting the pointwise arguments (the semi-discrete operator is linear in the stencil weights, so D acts on coefficients exactly as on the field). Let v_i = ∂u/∂θ_i, w_ij = ∂²u/∂θ_i∂θ_j, z_ijk = ∂³u/∂θ_i∂θ_j∂θ_k. Differentiating the ODE:

- **Order 1** (p members): dv_i/dt = L(u) v_i, with L(u) = ∂F/∂U evaluated at the fiducial, applied to (v_i, Dv_i, D²v_i); v_i(0) = ∂u0/∂θ_i.
- **Order 2** (p(p+1)/2 members): dw_ij/dt = L(u) w_ij + F''(u)[v_i, v_j]; w_ij(0) = ∂²u0/∂θ_i∂θ_j.
- **Order 3** (p(p+1)(p+2)/6 members): dz_ijk/dt = L(u) z_ijk + F''(u)[v_i, w_jk] + F''(u)[v_j, w_ik] + F''(u)[v_k, w_ij] + F'''(u)[v_i, v_j, v_k]; z_ijk(0) = ∂³u0/∂θ_i∂θ_j∂θ_k.

Key structural fact: **every order shares the same pointwise coefficient block L(u)** (a dense-per-point matrix over the ~24 evolved variables and their first/second derivatives, depending only on the fiducial); only the *source terms* differ. So the cost of a member relative to the nonlinear RHS is dominated by applying L (and for orders ≥ 2 the contractions F'', F''') at each grid point — the same stencil derivatives, the same unzip geometry, the same ghost exchange, the same remesh, one RK driver.

Observables: h = O(u) is also nonlinear (psi4 is quadratic-plus in the fields and their derivatives; mode decomposition is linear). Jets of h follow by the same rules: h_i = O'[v_i], h_ij = O'[w_ij] + O''[v_i, v_j], etc. Log-amplitude and phase of the complex mode H = A e^{iφ}: ∂_i ln H = H_i/H, so ∂_i ln A = Re(H_i/H), ∂_i φ = Im(H_i/H); second derivatives from ∂_ij ln H = H_ij/H − (H_i/H)(H_j/H). Verify against FD of amplitude and (unwrapped) phase.

## 3. Member counts and the univariate (jet) alternative

| p | order-1 members | order-2 | order-3 | central-FD gradient (2p+1 runs) | central-FD Hessian (2p²+1 runs) |
|---|---|---|---|---|---|
| 3 | 3 | 6 | 10 | 7 | 19 |
| 8 | 8 | 36 | 120 | 17 | 129 |

Full third order is only needed along a few directions (error estimator / trust radius), so use **univariate Taylor-mode jets**: for a direction d, propagate the degree-k truncated Taylor polynomial u(t; θ0 + s d) = Σ_k s^k u_k/k!. One jet of degree 3 along d costs three coupled linear-in-coefficient members (orders 1, 2, 3 *along d*), reusing L(u). Mixed derivatives of order 2 are recovered by polarization, ∂_ij h = ½[∂²_{d_i+d_j} h − ∂²_{d_i} h − ∂²_{d_j} h] (p(p+1)/2 jets of degree 2 ≡ the p(p+1)/2 order-2 members, now with a *shared* order-1 stage). For order 3 along r chosen directions: r jets of degree 3 (3r member solves in total, shared L), instead of r(r+1)(r+2)/6.

Informed-subspace reduction (B5): take the first-order Jacobian at the fiducial and the SNR-30 noise model, form the eigen-decomposition of the prior-preconditioned Fisher matrix, keep r informed directions; second order then needs r(r+1)/2 members (r = 2–4 ⇒ 3–10), third order r jets.

## 4. Cost model (to be replaced by measurements in B4)

Let N be the cost of one nonlinear RHS evaluation on the shared mesh (the PT_ASYNC FP64 kernel is ~37% of a step's GPU time; the whole step also pays unzip 25%, CAHD 18.5%, boundary 13%, remesh host).
- A tangent member applies L(u) to (v, Dv, D²v): same derivative stencils (6th order, 3 directions) on the member's fields, plus a pointwise contraction with the cached coefficient block. Model estimate: ≈ 1–1.5 N per order-1 member when the coefficient block is recomputed per point and stage and amortized over K members; order-2 members add the F'' contraction ≈ +30–60% (model). Ensemble sharing of unzip geometry/ghost exchange/remesh makes the per-member cost *below* the cost of an independent run by the unzip/remesh/launch fraction (≥ 30–45% of an independent solve today).
- Therefore (model, to be measured): J at p = 3 ≈ 1 forward + 3 members ≈ 3–4 forward-solve equivalents *or less* with sharing; the p = 3 order-2 block adds 6 members; p = 8 order 2 adds 36 (≈ 36–50 equivalents) vs 129 forward runs for central-FD Hessian — and FD Hessians carry the ~1e-9 noise floor and need an ε plateau study per component. Reported numbers must state: measured vs modeled, resolution, window.

## 5. Pipeline stages to differentiate (B2) — and what is frozen

1. **TwoPunctures initial data.** Tangents of u0(θ) are *linear elliptic solves with the Newton Jacobian at the converged solution* (reuse the operator and its factorization/preconditioner for all parameters and orders); second/third order add quadratic/cubic source terms evaluated from lower orders. Inputs: bare masses/momenta/spins (document the map to q, χ1z, χ2z and the orbit parameters).
2. **Mesh:** frozen to the fiducial's recorded sequence (replay). Remesh transfer and unzip/zip are linear and are applied coefficient-by-coefficient to the jets.
3. **RHS:** BSSN algebra + 6th-order stencils + KO dissipation: differentiate (B1 gives the exact discrete oracle via a truncated-Taylor scalar type; B4 generates GPU kernels from the SymPy source).
4. **Algebraic constraint enforcement** (det γ̃ = 1, trace Ã = 0): differentiate.
5. **Floors/clamps** (χ, α, …) and other branches: *frozen masks from the fiducial*; log how often a floor is active; an active floor in a region that influences the waveform is a non-differentiable point and must be reported (user stop condition).
6. **CAHD, slow-start lapse, gauge with coordinate-dependent η:** differentiate.
7. **psi4 extraction and mode decomposition:** differentiate.
8. **BH tracking (puncture position feeding grid refinement and the extraction center):** must not feed back into the differentiated path — confirm that tracking only affects the (frozen) mesh/refinement, not F.
9. **Time stepping:** RK stages are linear in (u, F): tangents use the same stage structure; the stage values u_s of the nonlinear solution are needed for L(u_s) — store or recompute per stage (checkpoint the fiducial trajectory or run fiducial and members in lock-step, which the ensemble layout does).

## 6. Verification plan (B3, required before any tangent performance work)

- Taylor remainder test: |h(θ + εd) − Σ_{k≤K} ε^k h_k/k!| must scale as ε^{K+1} until roundoff, for K = 1, 2, 3, on short runs (cap 13 is fine for development).
- Agreement with FD on the ε-plateau, in the early window *and* on the inspiral window.
- q = 1 exchange-symmetry identities (e.g. J_{m1} = J_{m2}, H_{11} = H_{22}) for the derivative with respect to the two bare masses.
- Growth check: tangents are expected to grow secularly (order-k derivatives roughly like t^k through dephasing). Spurious exponential growth from gauge or puncture regions must be characterized (user stop condition) before any suppression is attempted.
- Chain-rule checks of d ln A and dφ from complex-mode jets against FD of amplitude and phase.

## 7. Implementation staging

- **B1 (reference, CPU):** compile the existing generated BSSN algebra with a truncated-Taylor scalar type of degree 1, 2, 3 in place of `double`; stencils, KO, unzip/zip, intergrid transfer and RK are linear and act coefficient by coefficient. No symbolic differentiation; gives exact discrete derivatives on small problems.
- **B4 (GPU):** generate tangent kernels from the SymPy source in stages; compute the background subexpressions and the coefficient block once per point and stage and apply them to all members of all orders; avoid the operator-overloaded jet type on the GPU if it explodes registers (measure ptxas regs/spills either way). Ensemble layout: fiducial + K members in one launch, one unzip geometry, one ghost exchange, one remesh. Order of delivery: order 1 (p = 3) → order 2 → order 3 only as univariate jets along 1–3 directions. FP32 member storage only after FP64 passes B3.
- Existing groundwork in the repo: `BSSN_GR/cuda/src/tl_*_test.cu`, `bssn_tl_stub.cuh`, `CodeGen/bssn_tl_gauge.py` (standalone harnesses, 16/16 symbolic gates through gauge/χ/γ̃00/K/Ã00/Γ̃/lapse/toy-Ricci; not integrated, no full Ricci/C³ coefficient block, no driver mode in BSSNCtxGPU). The new design (coefficient block shared by all orders) should replace the per-variable stubs rather than extend them blindly.

## 8. Open questions for the first implementation step
- Memory: each member stores a full state (24 vars × unzipped padded blocks): at ~35k–90k elements a state is ~48–125 GB unzipped; K members multiply this ⇒ multi-GPU (members distributed over ranks sharing the mesh, or unzipped scratch reuse per block-batch) is probably required at production resolution. Decide the storage strategy (store only the zipped member states and recompute unzip per batch) before coding.
- Whether to propagate jets of observables (psi4 modes) on the fly or store member fields on the extraction shells only.
