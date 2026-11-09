# Open task queue (work in order; GPU only when the task needs it)

Do **not** run filler ETS loops. When a GPU task finishes, start the next item that needs the alloc. Login-side coding can proceed while waiting for PD nodes.

**Batching doctrine:** see `BATCHING.md`. Bank on K/CTA latency hiding + pack; do **not** sell FP64 TLM GEMM as speedup; POD/DEIM rejected; FP32 tangents only after plateau test.

| # | Task | Needs GPU? | Status |
|---|------|------------|--------|
| 1–16 | Tier 3.1 gauge→χ→gt→K→At TL gates + 3.2 Richardson | mixed | **done** (see prior rows in git history / TIER3_TL_PLAN) |
| 17 | Tier 3.3 sep-cost scout (reused TPID) | yes | **provisional** (`TIER33_ORBITS.md`) |
| **B0** | **2.1 NCU on `eval_rhs3_pt_async`** (regs/spills/occ) | yes (short) | **next** |
| **B1** | **2.1 staged algebra** (bit-id) if NCU shows reg headroom | mixed | queued (after B0) |
| **B2** | **K members/CTA interleave** — speedup/member vs K @ 64/256/2048 | yes | queued (after B1) |
| **B3** | Pack K ghost msgs + one RHS launch; measure fine-level | yes | queued (after B2) |
| 16b | Gt TL + Ricci/lap pieces; fused emit | mixed | parked (behind B0–B3) |
| 18 | Tier 3.3: re-TPID per sep / TARGET_ORBITS | mixed | parked |
| T_fp32 | FP32-tangent plateau vs Tier 1.5 FD plot | yes | later (after B2; not first) |

Automation: when a new idev goes `R`, run the **next queued GPU task**, not a keep-alive cycle.
