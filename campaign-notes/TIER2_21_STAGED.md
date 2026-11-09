# Tier 2.1 — staged algebra

## Prior art (parked)

`BSSN_GR/cuda/scripts/gen_gauge6_point.py` already prototyped deferring DD seconds until before K:

> bit-identical but **slower** on A100 (0.157s vs 0.137s) — the 11-var reload costs more than the live-range win. Keep unstaged as default.

`HANDOFF_NEXT.md` C1 still lists staged algebra as a candidate from NCU (L1TEX long-scoreboard, 56 regs, 0 spills → target ≤46 regs).

## Status on GB200 / PT_ASYNC

Not re-measured yet under CUDA-aware + PT_ASYNC on Horizon. Unzip/list-infra took priority and is **no-go**.

## Plan when revisiting

1. NCU on `eval_rhs3_pt_async` @ q8/4GPU: regs, spills, achieved occ, scoreboard.
2. Only if spills or reg pressure show headroom: regenerate staged algebra from SymPy (`CodeGen/bssn_manually_staged.py` / `bssn.py --staged_type staged`) behind an env flag.
3. A/B ETS + bit-id vs unstaged.

Until NCU says otherwise, treat 2.1 as **low priority** vs hang-path unzip / 2-node scale / TL (3.1).
