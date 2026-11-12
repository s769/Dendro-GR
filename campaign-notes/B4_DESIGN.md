# B4 design: production GPU tangent (Taylor-mode) kernels  -- DRAFT 2026-11-12 (written unattended; every number marked [est] is an estimate, [meas] measured)

## What is already verified (B1-B2, CPU, exact discrete derivatives through order 3, remainder slopes 1/2/3/4)
RHS algebra (SSL+CAHD+RIT eta, KO), RK4 stages + Sommerfeld boundary faces + enforce (det/trace/floors as frozen masks), TwoPunctures tangent (order 1: m+-, P, S), psi4/constraints. Linear parts (stencils, KO, BC, zip/unzip, ghost exchange, remesh transfer under frozen mesh, spherical-harmonic extraction) act per Taylor coefficient.

## Structure that the GPU implementation must preserve
1. Everything linear is applied independently to every member (coefficient field). Everything pointwise nonlinear is the generated algebra evaluated in a jet type. So the GPU code = existing production kernels launched on member arrays + ONE new pointwise kernel family (RHS, enforce, psi4) in jet arithmetic.
2. Univariate jets (direction d in parameter space): coefficients c0 (fiducial), c1..cK. Order-1 gradient for p parameters: p members each K=1 jet with c1 = tangent, sharing c0 (the fiducial trajectory is computed once; only L(u)-type coefficient blocks are shared). Order 2 along direction d needs K=2 jets; mixed second derivatives by polarization (p(p+1)/2 jets, or (p^2+p)/2 -> see TAYLOR_HIERARCHY.md member-count table).

## Layout (proposal)
- Member dimension is the slowest index of each state array: u[member][var][block][13^3]. One launch handles all members (grid.z = member) so unzip/ghost-exchange/zip/remesh cost is amortised in launch overhead and mesh metadata (indices, offsets) is shared.
- The fiducial (c0) is NOT replicated per member; members store only their c1..cK coefficient arrays; the pointwise kernel loads c0 once and c_j per member.
- Memory [est]: unzipped padded state per member-copy = 24 vars x 2197 pts x N_blocks x 8 B = 27.9 GB at cap-14 peak (66,032 blocks) and 36.6 GB at cap-15 peak (86,836 blocks). One production run needs several copies (RK stages, unzip in/out, k-arrays; footprint not yet measured -- TODO read nvidia-smi / allocator stats on a cap-14 run). If a production run at cap 14 needs ~3-4 copies, an order-1 p=3 ensemble (fiducial + 3 tangents) needs ~4x that, i.e. may exceed one 192 GB GPU: then use 4 GPUs (member parallel across ranks, no ghost exchange between members) -- member-level parallelism is embarrassingly parallel except the shared fiducial c0, which each rank recomputes (cost 1x) or receives.
- Peak memory is the gating item for K>1; FP32 tangent storage is only allowed after FP64 passes (user rule).

## Kernel generation
- Reuse `BSSN_GR/taylor/gen_taylor_eqs.py` / `gen_taylor_psi4.py` output but emit CUDA: device `Tay<K>` with fully unrolled recurrences; ptxas register/spill report is the first deliverable (850 statements x (K+1) live doubles: expect heavy spilling at K=3; strategy ladder: (a) compile-time K, one kernel per K; (b) split the algebra into the same stages production uses (CAHD separate kernel already exists: `gpu_apply_cahd`), (c) if spills dominate, generate per-order code with SymPy CSE directly for the first-order tangent (L(u)v form) instead of generic jets.
- Order 1 must be implemented as the Frechet derivative: dRHS = L(u) v. Cheapest implementation = the K=1 jet kernel.
- Verification harness for every kernel: the existing CPU `bssnTaylorRef`/`bssnTaylorStep`/`bssnTaylorPsi4` become the oracle (same random block data fed to the GPU kernel; require agreement ~1e-14 and the same remainder slopes against the production GPU double RHS at displaced eps).

## Cost model (to be replaced by measurements)
- CPU pointwise algebra per grid point relative to K=0: measured by `TAYLOR_TIMING=1 ./bssnTaylorRef` (result will be appended to STATE when run).
- Expectation [est]: a K-jet costs between (K+1) and (K+1)(K+2)/2 times the plain algebra (cross terms in products), stencils cost (K+1)x exactly. So order-1 jets: <= ~3x per member, i.e. p members ~ 1x (fiducial) + p x 2-3x of one forward solve of GPU work if not memory/bandwidth-bound; since the production kernels are bandwidth-bound (unzip 25%, CAHD 18.5%, boundary RHS 13% of time) the tangent members may cost closer to p x 1.5-2x. The real number comes from B4 profiling.
## Open design questions (need user input or measurement)
1. Parameter count p and which parameters (affects ensemble size and memory).
2. Multi-GPU: member-parallel vs domain-parallel for the ensemble at cap >= 15.
3. Second-order: univariate jets + polarization versus direct bivariate (K=2 in 2 variables) -- count of members vs complexity.
