/**
 * @file bssn_tl_stub.cuh
 * @brief Tier 3.1 tangent-linear RHS — scaffold only (not linked yet).
 *
 * Plan: TIER3_TL_PLAN.md / campaign-notes/TIER3_TL_PLAN.md
 *
 * Intended API (FP64):
 *   - Background unzipped state u (+ Du/DDu from existing PT path)
 *   - K tangent vectors v[k] (same block list; ensemble grid.y reuse)
 *   - Output Jv[k] = (d RHS / d u) · v[k]
 *
 * First implementable slice: gauge sector (alpha, beta) only, bit-check vs FD
 * on frozen-mesh q1. Full BSSN follows once harness works.
 */
#pragma once

namespace device {
namespace tl {

// Placeholder — do not call from production RHS.
template <typename T>
struct TlWorkspace {
  T* v_uz;   // [K * dof_unzip] tangent unzipped
  T* jv_uz;  // [K * dof_unzip] output
  unsigned int K;
};

template <typename T>
inline void bssnrhs_tl_gauge_stub(TlWorkspace<T>& /*ws*/) {
  // Not implemented. See TIER3_TL_PLAN.md work package 1–2.
}

}  // namespace tl
}  // namespace device
