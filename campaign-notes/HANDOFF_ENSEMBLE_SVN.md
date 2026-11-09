# Handoff: forward solver work, reprioritized for the ensemble/SVN campaign

Context: the end use is a Bayesian inversion where each outer iteration runs O(100) BBH simulations, each of which is a small ensemble (1 fiducial + 2r perturbed members, r ≈ 2–3) on a shared, frozen mesh. The critical path is single-simulation time-to-solution at q ≈ 8–16, then ensemble throughput, then derivative consistency. Precision stays FP64. PT_F32SU is parked as a validated hedge; do not spend more time on FP32 unless asked.

**Status:** Tier 0 = **NO-GO GPU LTS**. Tier 1 mostly done (1.5 dual-res FD caution). Tier 2: CUDA-aware MPI on (q8/4GPU 3.03→2.13s, eff 0.56); unzip hang/nh split **no-go** (hang 70–99% of active; `TIER2_UNZIP.md`). Next: hang-path/unzip overlap, 2.1 staged algebra, 2.5 2-node, 3.1 TL. Production: `DENDRO_RHS_PT_ASYNC=1` + `MPIX_CUDA_AWARE_SUPPORT=ON`.

## Tier 0 — Decide the gate: local time stepping on the GPU

Everything downstream depends on whether high-q single simulations can be made cheap, and that is step count, not RHS cost.

0.1 Audit Dendro-5.01's ENUTS (explicit non-uniform timestepping) against `BSSNCtxGPU`. Report: what ENUTS needs from a context (per-level RHS over block subsets, inter-level interpolation in time, partial unzip/zip), which of those the GPU context already provides, and which are CPU-only. Produce a gap list with an effort estimate per item.

0.2 Measure the payoff before building: for a real q = 4 and q = 8 mesh, dump the block-count-per-level histogram and compute the theoretical step-work ratio Σ_l N_l 2^l / (N_total 2^L_max). That number is the ceiling on what LTS buys and decides whether it's a 3× or a 15× item.

0.3 Characterize the small-nblocks regime now, since LTS lives there: `rhsgpu_test` at 32, 64, 128, 256 blocks, with nsys launch gaps and achieved occupancy. This is the baseline the ensemble work in Tier 1 has to fix.

Exit criterion: a go/no-go on GPU LTS with a scoped plan. If no-go, say so and why; the inversion plan then rests on hybridization and resolution instead.

## Tier 1 — Ensemble dimension and frozen-mesh replay

1.1 Frozen-mesh replay. Record the refinement history of a fiducial run (block lists per remesh event, remesh times) and add a mode that replays it exactly in a second run with different initial data. No `isReMeshWAMR` decisions in replay mode. Validate: fiducial replayed against itself is bit-identical.

1.2 Deterministic initial data. Make the TwoPunctures parameter → initial data path reproducible and parametrized by bare quantities (bare masses, momenta, spins); any eccentricity-reduction or mass-correction iteration must be either frozen (fixed iteration count and inputs) or moved out of the differentiated path. Document the exact parameter vector the inversion will differentiate.

1.3 Ensemble dimension in the GPU context. Add K as a leading dimension on the unzipped vectors and the RK state, with one block list shared by all members. RHS kernel: either blocks × K CTAs in one launch (first version, trivial) or K members per CTA sharing the block geometry (second version, better smem/launch amortization at small nblocks). Measure RHS throughput per member versus K at 64, 256, 2048 blocks. Target: at 64 blocks, K = 8 members should run in well under 8× the single-member time; this is what makes LTS fine levels and perturbation runs efficient at once.

1.4 Bit reproducibility across runs and across ensemble slots. Same inputs → same bits, regardless of K or which slot a member occupies. Eliminate atomics in reductions (the constraint-enforcement min-det atomic, any norm reductions) or make them order-fixed.

1.5 Jacobian convergence test. On a q = 1 short run with frozen mesh, compute finite-difference waveform derivatives w.r.t. one bare parameter at step sizes spanning two orders of magnitude, at two resolutions. Plot the derivative versus step; identify the plateau. This single plot decides whether FD is sufficient or the tangent-linear model (Tier 3) is required.

## Tier 2 — FP64 RHS and full-step work (LTS/ensemble in mind)

2.1 Staged algebra from the SymPy source to cut live ranges and spills (bit-identical).
2.2 Fuse zip into the RHS kernel (write interior points straight to the nodal vector).
2.3 Launch hygiene: merge the hybrid face/interior double launch; compact block lists; CUDA graphs for the RK stage sequence. This matters more now because LTS and ensembles both increase launch count.
2.4 Full-step profile at q = 4 on 4 GPUs with `__PROFILE_CTX__`: RHS / unzip / zip / ghost exchange / remesh / RK update, and the same for a q = 8 mesh even if only for a few steps. Replace the extrapolated wall-time table with these.
2.5 Single-simulation strong scaling across the 4-GPU node (and 2 nodes) at q = 8. High-q time-to-solution is the critical path, and a single simulation needs to use at least a node efficiently. Report parallel efficiency and where it breaks (communication vs load imbalance at fine levels).
2.6 19³ blocks, only after 2.4 shows what fraction of points could coalesce.

## Tier 3 — Differentiation and fidelity knobs (start only after Tier 1.5)

3.1 Tangent-linear RHS via codegen if FD plateaus are poor: differentiate the symbolic BSSN RHS w.r.t. state and derivatives, emit a linearized RHS kernel that takes the background state and K tangent vectors, share stencil work across members. Verify against FD on the plateau from 1.5. Expect growing gauge modes near punctures; test renormalization of the tangent if needed.

3.2 Resolution as a clean knob. Ensure a run at resolution h and one at h/1.5 (or 2) differ only in the refinement depth parameter, with identical frozen-mesh topology otherwise, so Richardson differences are meaningful. Produce one Richardson pair at q = 4 as a sample of the discretization-error model.

3.3 Hybridization length. Expose "number of orbits before merger" as a controlled initial-data parameter and measure cost versus orbits at q = 8; this is a physics decision but the solver needs to make it a knob.

## What to stop doing

FP32 variants beyond keeping PT_F32SU compiling and running its existing smoke. Tensor-core experiments on the *nonlinear* RHS. Further `launch_bounds` sweeps on the stock kernel. POD/DEIM of the tangent. Selling FP64 TLM GEMM as an acceleration (exactness only — see `BATCHING.md`).

## Batching order (bankable)

1. 2.1 staged algebra (regs) → 2. K/CTA FP64 interleave speedup curve → 3. pack ghosts/launches → 4. FP32-tangent plateau (later). Details: `BATCHING.md`.

## Reporting

Per item: what changed, numbers in the same format as GB200_RESULTS.md, and for anything touching numerics the bit-reproducibility check and the 1.5-style convergence plot. Negative results stay in the negatives list with a reason.

## Log

| Item | Status | Artifact |
|------|--------|----------|
| 0.1 ENUTS×GPU audit | **done** | `ENUTS_GPU_AUDIT.md` |
| 0.2 LTS payoff hist | **provisional** | default `LTS_TS_OFFSET=4` → ~1.6×; dump queued in `run_tier0.sh` |
| 0.3 small-nblocks nsys | **provisional** | launch-bound at ≤128; fresh nsys queued |
| **Tier 0 gate** | **NO-GO GPU LTS** | `TIER0_LTS_GATE.md` — re-cost on hybridization + resolution + ensemble |
| Prior FP64 RHS notes | parked under Tier 2 | `HANDOFF_NEXT.md` (superseded as primary) |
| 1.1 frozen-mesh replay | **done** | flags+ReMesh; fiducial `max\|ΔC_HAM\|=1e-18` on c101-014 (`TIER1_MESH_REPLAY.md`) |
| 1.2 deterministic TPID | **done** | bare←`BSSN_BH*.MASS`; dual-tpid bit-id q1+q4 (`TIER1_TPID.md`) |
| 1.3 ensemble K (v1) | **done** | `grid.y=K`; @64bl K=8 → 5.7× single (`TIER1_ENSEMBLE.md`); K/CTA still open |
| 1.4 bit-id | **mostly done** | min-det ordered fold bit-id; residual ~1e-18 in L2 norms (`TIER1_BITID.md`) |
| 1.nf non-fiducial init | **done** | skip init WAMR under replay; record/replay OK on c101-014 |
| 1.5 FD Jacobian | **done (gate caution)** | short C_HAM plateau OK; dual-res coarse −1.662e-6 vs fine −6.153e-5 — no dual-res agreement (`TIER1_FD.md`) |
| 2.4 short q4/q8 4GPU | **done** | q4 ETS≈4.42s ceiling 10.7×; q8 ETS≈8.94s ceiling 14.3× (`tier0/q{4,8}_*`) |
| 1.3 v2 K/CTA | **done (no-go)** | CTA-serial K-share slower than v1 at 64bl (8.1× vs 5.7× single); keep v1 (`TIER1_ENSEMBLE_V2.md`) |
| 2.5 q8 strong scale | **done (weak)** | 1→4 GPU ETS 4.71→3.03s (eff 0.39); prior 4GPU nums suspect CVD=0 (`TIER2_SCALE.md`) |
| 2.4 nsys q8 4GPU | **done** | RHS 40% / unzip 28% / stock-bdy 19%; H2D+D2H dominate memops (`TIER2_PROFILE.md`) |
| 2.5 q4 strong scale | **done (weak)** | same eff≈0.39 @4GPU as q8 (`TIER2_SCALE.md`) |
| 2.2 zip-fuse | **parked** | zip only 1.4% kernel time (`TIER2_NEXT.md`) |
| 2.3 compact bdy | **prototype** | 3–10% blocks are bdy; host-compact ETS-neutral; need MeshGPU-persistent list (`TIER2_NEXT.md`) |
| 2.5 CUDA-aware MPI | **done** | `-DMPIX_CUDA_AWARE_SUPPORT=ON`: q8/4GPU 3.03→2.13s, eff 0.39→0.56; q4 1.45→1.19 (`TIER2_SCALE.md`) |
| 2.3 device compact | **no-go ETS** | device compact 2.118 vs 2.121; stock 19% is real bdy work (`TIER2_NEXT.md`) |
| 2.unzip hang/nh split | **no-go ETS** | hang 70–99% of active; dual +0–2%; active-skip −0.5% noise (`TIER2_UNZIP.md`) |
| 2.5 2-node q8 | **done** | 4→8 GPU 2.145→1.676s (1.28×, eff 0.64); 1→8 eff 0.35 (`TIER2_SCALE.md`) |
| 2.1 staged algebra | **parked** | prior A100 proto slower; need GB200 NCU first (`TIER2_21_STAGED.md`) |
| 2.unzip batch-sync defer | **ETS-neutral** | default deferred; prod had `ASYNC_COMM_K=1` so no overlap (`TIER2_UNZIP.md`) |
| 2.unzip async_k sweep | **no-go** | K=1 best (2.13s); K↑ monotonically worse to 2.73 @K=8 (`tier0/tier2_async_k_sweep.log`) |
