# Batching on GB200 — what pays (and what doesn’t)

Source: campaign note 2026-11-08. This overrides any agent pitch that TLM/GEMM is an FP64 speedup.

## Rule-outs

| idea | verdict |
|------|---------|
| Tensor cores for K nonlinear members | **no** — batching grows the *output* dim; stencil reduction stays 13 (or 7). MMA unused. |
| FP64 TLM GEMM as acceleration | **no** — K=16 costs ~26 RHS-eq vs 16 FD; value is exactness / step-size, not speed. Dense GEMM also wastes ~2–3× on sparse C. |
| POD/DEIM of the tangent | **reject** — slow Kolmogorov n-width; phase accuracy over ~10⁶ steps fails. Mention as considered/rejected in papers only. |
| Full-state FP32 | already **dangerous** (u≈1+ε); do not reopen. |

## Bankable (boring) wins — do these

1. **K members/CTA interleaving (FP64)** after staged algebra cuts live ranges. Stall-bound kernel (~10% FP64 peak, <20% HBM): K independent chains hide latency. Target **2–3× per member at K=4–8**, limited by regs (56×K won’t fit without staging). Smem ~K×17.6 KB/var → K≤8 one var at a time on GB200.
2. **Message + launch packing** — one ghost exchange / one RHS launch for K members. Dominates at fine LTS (few blocks). Makes the ensemble defensibly coupled.
3. **Shared geometry** — block lists, unzip maps, interp coeffs, fiducial traj once per ensemble.
4. **Batched TwoPunctures** — spectral elliptic, real DMMA shape; small share of wall → tidy win, not a headline.

## Contingent experiment (not first)

**FP32 tangents on FP64 background** (rescale δu to O(1) each step). Same Jacobian-convergence plot as Tier 1.5. If it holds: C(u) build in FP64, GEMM via 3×TF32 → ~1.5–2× on perturbation members only at K=16 after substantial engineering. Measure TLM-vs-FD accounting; don’t estimate.

TLM shape reminder: per point δF = C(u)·φ(δu), C is 24×162 (~30–40% dense, ~31 KB — rebuild every RK stage), K members → (24×162)×(162×K).

## Agent ordering

1. **2.1 staged algebra** (bit-id) → free regs / kill local spills for K/CTA.
2. **K/CTA interleave** — per-member speedup vs K curve (64 / 256 / 2048 blocks). Keep **one** var tile in smem (NCU: occ already smem-bound at 17.6 KB).
3. **Pack ghosts + launches** — measure at fine-level block counts.
4. **FP32-tangent plateau** alongside FP64 FD (after harnesses); only then TLM+TF32 prototype.

Do **not** sell TLM as an acceleration story in FP64.

## GB200 PT_ASYNC facts (B0 NCU, 2026-11-08)

- 56 regs/thread; **local-mem 84% of L1TEX sectors**; spill requests huge.
- Occupancy **17%**, limited by **smem** (not regs). SM throughput 13.5%; ~7% FP64 peak; L1TEX scoreboard ~53% of issue stalls.
- Dropping smem (PT_GMEM) **hurts** ETS (+51%). F32SU −8% ETS pending bit-id. Staging must cut *local spills / live ranges*, not delete the shared tile.