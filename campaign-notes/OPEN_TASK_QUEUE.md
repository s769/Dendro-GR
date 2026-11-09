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

Automation: when a new idev goes `R`, run the **next queued GPU task**, not a keep-alive cycle.
