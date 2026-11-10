# Open task queue (work in order; GPU only when the task needs it)

Do **not** run filler ETS loops. When a GPU task finishes, start the next item that needs the alloc. Login-side coding can proceed while waiting for PD nodes.

**Batching doctrine:** see `BATCHING.md`. Bank on K/CTA latency hiding + pack; do **not** sell FP64 TLM GEMM as speedup; POD/DEIM rejected; FP32 tangents only after plateau test.

| # | Task | Needs GPU? | Status |
|---|------|------------|--------|
| 1–16 | Tier 3.1 gauge→χ→gt→K→At TL gates + 3.2 Richardson | mixed | **done** (see prior rows in git history / TIER3_TL_PLAN) |
| 17 | Tier 3.3 sep-cost scout (reused TPID) | yes | **provisional** (`TIER33_ORBITS.md`) |
| **B0** | **2.1 NCU on `eval_rhs3_pt_async`** (regs/spills/occ) | yes (short) | **done** (56 regs; **local spill heavy**; occ 17% **smem-limited**) |
| **B1a** | PT_GMEM / F32SU A/B vs ASYNC | yes | **done** (GMEM **NO-GO**; F32SU −8% ETS, **not** C_HAM bit-id) |
| **B1-defer** | deferred DD seconds in PT_ASYNC | yes | **NO-GO** (+7.7% ETS on GB200; same as A100) |
| **B1-scoped** | section-scoped PT (restencil late stages) | yes | **NO-GO** (+123% ETS) |
| **B1-lb** | `launch_bounds(343,3)→(343,1)` on PT_ASYNC | yes | **done** (~**−15%** ETS; keep) |
| **B1** | **2.1 live-range/CSE without restencil** (for K/CTA headroom) | mixed | **CSE NO-GO**; **while-hot keep opt-in** (q8 −0.7%, q1 −0.23%; `tier21_whilehot_ab.md`) |
| **B2** | **K members/CTA interleave** — speedup/member vs K @ 64/256/2048 | yes | **baseline done** (`tier0/tier21_ensK_baseline.md`; small-nblocks only) |
| **B3** | Pack K ghost msgs + one RHS launch; measure fine-level | yes | **RHS done** (2.89×@32bl); pack API in; **q8+q1 pack NO-GO** (0.5–0.9×) — park pack wire until few-block LTS |
| 16b | Gt TL + Ricci/lap pieces; fused emit | mixed | **Gt+lap+toyRicci+fused emit 16/16**; full Ricci/C³ next |
| 18 | Tier 3.3: re-TPID per sep / TARGET_ORBITS | mixed | **done scaffold+re-TPID N=3/5/8 + short cost**; longer cost next |
| T_fp32 | FP32-tangent plateau vs Tier 1.5 FD plot | yes | later (after B2; not first) |

Automation: when a new idev goes `R`, run the **next queued GPU task**, not a keep-alive cycle.

## Update 2026-11-09 (campaign continuation)
| item | status |
|------|--------|
| B2 K/CTA interleave | **closed** (measured ~1.0-1.15x at production block counts; see BATCHING.md) |
| B3 pack K ghosts | closed (no-go at these sizes) |
| Remesh: GPU BH_WAMR + device WAMR + block-setup O(N^2) fix | **done, default ON** (adaptive production-depth q8 380.8 -> 190.4 s) |
| Extraction (shell-only, OMP, cached sampler) | done (opt-in `DENDRO_GW_SHELL_ONLY=1`) |
| init_grid initial-data evaluation (~580 s/simulation host, single thread) | OpenMP version implemented (`DENDRO_INIT_OMP`, default on), **timing pending a node** |
| var-batched unzip CTA | idea, not started |
| FP32-tangent plateau test | later |
| Tier 3.3 q8 orbits-vs-cost | not started |
| fine-resolution waveform FD | not started |
| GPU-vs-CPU long-run physics comparison with BH_WAMR | not started |
