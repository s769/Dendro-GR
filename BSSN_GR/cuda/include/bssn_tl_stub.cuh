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

/**
 * Linearization of radiative_bc_pt for a scalar (e.g. lapse).
 * Background: f_rhs = -r^{-1}(x f_x + y f_y + z f_z + falloff*(f - asym))
 * Tangent:    Jf    = -r^{-1}(x v_x + y v_y + z v_z + falloff*v)
 * (geometry / falloff / asym fixed). Verified by tl_gauge_radbc_test.
 */
template <typename T>
inline T radiative_bc_tl(T x, T y, T z, T vx, T vy, T vz, T v, T falloff) {
  const T inv_r = T(1) / sqrt(x * x + y * y + z * z);
  return -inv_r * (x * vx + y * vy + z * vz + falloff * v);
}

}  // namespace tl
}  // namespace device
