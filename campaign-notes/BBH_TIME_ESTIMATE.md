# Full BBH merger wall-time estimates (Horizon GB200)

**All tables below are extrapolations from A100 paper walls × GB200 RHS microbench ratios until `HANDOFF_NEXT.md` item D (timed q=4 evolve) exists.**

Sources: Dendro-GR SC’22 / ET-school slides (Lonestar6 A100 results); our GB200 RHS microbenches.

## Published A100 reference (4–8 GPUs, separation 8M)

| q | GPUs | timesteps | Wall time | notes |
|---|-----:|----------:|----------:|-------|
| 1 | 4×A100 | 183K | **87 h** | |
| 2 | 4×A100 | 252K | **96 h** | |
| 4 | 4×A100 | 506K | **129 h** | |
| 8 | 8×A100 | 4M | **388 h** (~16 d) | |

Strong scaling on A100 stayed good to ~8 GPUs (eff ~0.64 at 8). Weak scaling to 229K CPU cores on Frontera.

## How cost grows with mass ratio

From the same slides: timesteps scale roughly as **q↑ → finer Δx near small BH → smaller CFL dt → more steps**. Rough step counts: q1 ~8e4–2e5, q4 ~2e5–5e5, q16 ~1e6, q64 ~2e7. Wall time is not linear in q because AMR DOFs and remesh also grow.

## Extrapolating to Horizon GB200

Single-GPU RHS microbench vs A100 baseline (~0.171 s):

| Path | GB200 time | vs A100 base |
|------|----------:|-------------:|
| stock | ~0.10–0.12 s | ~1.4–1.7× |
| PT_ASYNC / PT_DB | ~0.060–0.074 s | ~2.3–2.9× |
| F32C (validated) | ~0.056–0.061 s | ~2.8–3.0× |
| **PT_F32SU (validated)** | **~0.050 s** | **~3.4×** |

**Caveats:** microbench is dense interior blocks only. Production splits RHS (~70%) / unzip (~20%) / rest. High-q adds remesh + more face blocks (hybrid stock faces). End-to-end speedup is **less** than raw RHS — use **~1.7–2.5×** vs published A100 wall times with PT_F32SU (RHS-weighted), until a real q≥4 evolve is timed.

### Working estimates (same GPU *count* as paper)

| q | Paper (A100) | Horizon GB200 (PT_F32SU ~2.3×) | conservative 1.7× |
|---|-------------:|------------------------------:|------------------:|
| 1 (4 GPU) | 87 h | **~35–40 h** | ~50 h |
| 2 (4 GPU) | 96 h | **~40–45 h** | ~55–60 h |
| 4 (4 GPU) | 129 h | **~50–60 h** | ~75 h |
| 8 (8 GPU) | 388 h | **~150–170 h** (~6–7 d) | ~230 h |

With **4×GB200 per Horizon node** (paper often 1 GPU/node class), same GPU *count* fits in fewer nodes; strong-scaling efficiency from Dendro-GR papers (~0.89@4, ~0.64@8) still applies across nodes.

### If you scale out further

Paper strong-scaling efficiency ~0.89 @4 GPU, ~0.64 @8. Extrapolating loosely to 16–32 GB200s (not measured):

- **q=4 on 16 GPUs:** maybe **~1.5–2.5 days** if scaling holds moderately
- **q=16–64:** still dominated by step count (10⁶–10⁷); expect **weeks** even on tens of GPUs unless dt / AMR strategy improves

## High-q specific levers (this campaign)

1. RHS throughput (PT_ASYNC / PT_DB / F32C) — helps every step  
2. Remove hybrid double-launch on faces (port one-sided FD) — more face work at coarse/fine boundaries  
3. Remesh / unzip cost (not yet attacked) — grows with AMR depth  
4. Multi-GPU per Horizon node (4×GB200) — paper used 1 GPU/node class; we have 4/node to exploit

## MEASURED (2026-11-09, GB200, 4 GPU, production-depth q8, first 1,729 steps) — supersedes extrapolated step cost above
Pure stepping 0.044 s/step (mesh 18.7k elements, dt 1.0e-3). With production cadences and original code: 0.092 s/step (remesh test 26%, extraction 18%, checkpoints 3%).
After extraction fix: 0.077 s/step. Per-step cost grows with mesh size and shrinking dt later in the run, so this is NOT yet a full-run wall-time estimate. Details: `OPTIMIZATION_LOG.md`.
