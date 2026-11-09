# Open task queue (work in order; GPU only when the task needs it)

Do **not** run filler ETS loops. When a GPU task finishes, start the next item that needs the alloc. Login-side coding can proceed while waiting for PD nodes.

| # | Task | Needs GPU? | Status |
|---|------|------------|--------|
| 1 | Tier 3.1: host lapse TL vs FD harness | no | **done** (`tl_gauge_fd_test` PASS) |
| 2 | Tier 3.1: device PT-stencil TL for lapse + FD gate | yes (short) | **done** (`tl_gauge_device_test` PASS on c125) |
| 3 | Tier 3.1: real unzipped q1 block dump → TL vs FD | yes | **done** (`tl_gauge_q1_test` PASS; dump blk5 bflag=0) |
| 4 | Hang-path unzip: NCU + TB88 A/B | yes | **done** (NCU: uncoalesced gather; TB88 **NO-GO** +4.4%) |
| 5 | Frozen-mesh unzip reuse (ensemble) | yes | **provisional** (KSHARE ~10–15% vs serial-K; gather still dominates) |
| 6 | 2.1 staged algebra only after NCU says spills/regs matter | yes (ncu) | parked |
| 7 | Coalesced hang gather (sort-by-cg) | yes | **NO-GO** (+78% ETS); park / try CG-scatter later |
| 8 | Tier 3.1: radiative BC TL linearization | yes (short) | **done** (`tl_gauge_radbc_test` FACE+INTERIOR PASS) |
| 9 | Tier 3.1: expand TL to shift (β⁰) interior | yes (short) | **done** (`tl_shift_beta_test` PASS) |
| 10 | Tier 3.1: Γ-driver B⁰ TL (δB⁰ + δβ⁰) | yes (short) | **done** (`tl_Bdriver_test` PASS) |
| 11 | Tier 3.1: SymPy TL codegen (gauge→χ) + device χ gate | mixed | **done** (`bssn_tl_gauge.py` 7/7; `tl_chi_test` PASS) |
| 12 | Tier 3.2: dual-res Richardson pair at q=4 (maxdepth knob) | yes | **done** (`TIER32_RICHARDSON.md`) |
| 13 | Tier 3.1: SymPy TL gt00 scaffold + device gate | mixed | **done** |
| 14 | Tier 3.1: full tensor Lie gt00 (δgt0/δgt1/δα/δAt0) | mixed | **done** (`bssn_tl_gauge` 11/11; `tl_gt_test` PASS) |
| 15 | Tier 3.1: K TL (product/KO; lap(α) deferred) | yes (short) | **done** (`tl_K_test` PASS) |
| 16 | Tier 3.1: At00 algebraic+Lie TL (Ricci deferred) | yes (short) | **done** (`tl_At_test` PASS) |
| 16b | Tier 3.1: Gt TL + Ricci/lap pieces; fused emit | mixed | **next** |
| 17 | Tier 3.3: sep-cost scout (reused TPID; provisional) | yes | **provisional** (`TIER33_ORBITS.md`) |
| 18 | Tier 3.3: re-TPID per sep / TARGET_ORBITS → fair q8 cost | mixed | **next** |

Automation: when a new idev goes `R`, run the **next queued GPU task**, not a keep-alive cycle.
