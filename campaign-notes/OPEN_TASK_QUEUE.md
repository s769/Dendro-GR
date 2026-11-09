# Open task queue (work in order; GPU only when the task needs it)

Do **not** run filler ETS loops. When a GPU task finishes, start the next item that needs the alloc. Login-side coding can proceed while waiting for PD nodes.

| # | Task | Needs GPU? | Status |
|---|------|------------|--------|
| 1 | Tier 3.1: host lapse TL vs FD harness | no | **done** (`tl_gauge_fd_test` PASS) |
| 2 | Tier 3.1: device PT-stencil TL for lapse + FD gate | yes (short) | **done** (`tl_gauge_device_test` PASS on c125) |
| 3 | Tier 3.1: real unzipped q1 block dump → TL vs FD | yes | **next** |
| 4 | Hang-path unzip: next real lever (occupancy/smem) after ncu or targeted microbench | yes | queued |
| 5 | Frozen-mesh unzip reuse (ensemble) | yes | queued |
| 6 | 2.1 staged algebra only after NCU says spills/regs matter | yes (ncu) | parked |

Automation: when a new idev goes `R`, run the **next queued GPU task**, not a keep-alive cycle.
