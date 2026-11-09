/**
 * Point-wise + fused FD stencils for the BSSN RHS (interior / bflag==0).
 * Algebra only needs the thread's center point — no Du/DDu block passes.
 * Include from inside namespace device.
 */
#pragma once

#ifndef IDX_PT
#define IDX_PT(i, j, k, nx) ((i) + (nx) * ((j) + (nx) * (k)))
#endif

/** Issue async 8-byte copies of a 13³ block into shared; does not wait. */
template <unsigned int nx>
DEVICE_FUNC __forceinline__ void __ld_blk_var1_async_issue__(
    DEVICE_REAL* __restrict__ s_u, const DEVICE_REAL* __restrict__ var) {
    const DEVICE_INT tid =
        (GPUDevice::thread_id_z() * GPUDevice::block_dim_x() +
         GPUDevice::thread_id_y()) *
            GPUDevice::block_dim_y() +
        GPUDevice::thread_id_x();
    const DEVICE_INT n   = nx * nx * nx;
    const DEVICE_INT inc = GPUDevice::block_dim_x() * GPUDevice::block_dim_y() *
                           GPUDevice::block_dim_z();
    for (DEVICE_INT k = tid; k < n; k += inc) {
        unsigned long long smem_addr, gmem_addr;
        asm volatile("cvta.to.shared.u64 %0, %1;"
                     : "=l"(smem_addr)
                     : "l"(s_u + k));
        asm volatile("cvta.to.global.u64 %0, %1;"
                     : "=l"(gmem_addr)
                     : "l"(var + k));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :
                     : "l"(smem_addr), "l"(gmem_addr));
    }
    asm volatile("cp.async.commit_group;");
}

DEVICE_FUNC __forceinline__ void __ld_blk_var1_async_wait__() {
    asm volatile("cp.async.wait_group 0;");
    GPUDevice::sync_threads();
}

/** First + second + KO along +x. One ±3 read of su. */
template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_x3(
    const DEVICE_REAL* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dx,
    DEVICE_REAL* __restrict__ dx, DEVICE_REAL* __restrict__ dxx,
    DEVICE_REAL* __restrict__ kox) {
    const int g = IDX_PT(i, j, k, nx);
    const DEVICE_REAL sm3 = su[g - 3], sm2 = su[g - 2], sm1 = su[g - 1];
    const DEVICE_REAL s0  = su[g];
    const DEVICE_REAL sp1 = su[g + 1], sp2 = su[g + 2], sp3 = su[g + 3];
    if (dx)
        *dx = (-sm3 + 9.0 * sm2 - 45.0 * sm1 + 45.0 * sp1 - 9.0 * sp2 + sp3) *
              (inv_dx / 60.0);
    if (dxx)
        *dxx = (2.0 * sm3 - 27.0 * sm2 + 270.0 * sm1 - 490.0 * s0 + 270.0 * sp1 -
                27.0 * sp2 + 2.0 * sp3) *
               ((inv_dx * inv_dx) / 180.0);
    if (kox)
        *kox = (-sm3 + 6.0 * sm2 - 15.0 * sm1 + 20.0 * s0 - 15.0 * sp1 +
                6.0 * sp2 - sp3) *
               (-inv_dx / 64.0);
}

template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_y3(
    const DEVICE_REAL* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dy,
    DEVICE_REAL* __restrict__ dy, DEVICE_REAL* __restrict__ dyy,
    DEVICE_REAL* __restrict__ koy) {
    const int g = IDX_PT(i, j, k, nx);
    const DEVICE_REAL sm3 = su[g - 3 * nx], sm2 = su[g - 2 * nx],
                      sm1 = su[g - nx];
    const DEVICE_REAL s0  = su[g];
    const DEVICE_REAL sp1 = su[g + nx], sp2 = su[g + 2 * nx],
                      sp3 = su[g + 3 * nx];
    if (dy)
        *dy = (-sm3 + 9.0 * sm2 - 45.0 * sm1 + 45.0 * sp1 - 9.0 * sp2 + sp3) *
              (inv_dy / 60.0);
    if (dyy)
        *dyy = (2.0 * sm3 - 27.0 * sm2 + 270.0 * sm1 - 490.0 * s0 + 270.0 * sp1 -
                27.0 * sp2 + 2.0 * sp3) *
               ((inv_dy * inv_dy) / 180.0);
    if (koy)
        *koy = (-sm3 + 6.0 * sm2 - 15.0 * sm1 + 20.0 * s0 - 15.0 * sp1 +
                6.0 * sp2 - sp3) *
               (-inv_dy / 64.0);
}

template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_z3(
    const DEVICE_REAL* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dz,
    DEVICE_REAL* __restrict__ dz, DEVICE_REAL* __restrict__ dzz,
    DEVICE_REAL* __restrict__ koz) {
    const int nxy = nx * nx;
    const int g   = IDX_PT(i, j, k, nx);
    const DEVICE_REAL sm3 = su[g - 3 * nxy], sm2 = su[g - 2 * nxy],
                      sm1 = su[g - nxy];
    const DEVICE_REAL s0  = su[g];
    const DEVICE_REAL sp1 = su[g + nxy], sp2 = su[g + 2 * nxy],
                      sp3 = su[g + 3 * nxy];
    if (dz)
        *dz = (-sm3 + 9.0 * sm2 - 45.0 * sm1 + 45.0 * sp1 - 9.0 * sp2 + sp3) *
              (inv_dz / 60.0);
    if (dzz)
        *dzz = (2.0 * sm3 - 27.0 * sm2 + 270.0 * sm1 - 490.0 * s0 + 270.0 * sp1 -
                27.0 * sp2 + 2.0 * sp3) *
               ((inv_dz * inv_dz) / 180.0);
    if (koz)
        *koz = (-sm3 + 6.0 * sm2 - 15.0 * sm1 + 20.0 * s0 - 15.0 * sp1 +
                6.0 * sp2 - sp3) *
               (-inv_dz / 64.0);
}

/** Fused FD reading float shared tile; results promoted to DEVICE_REAL. */
template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_x3_f32su(
    const float* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dx,
    DEVICE_REAL* __restrict__ dx, DEVICE_REAL* __restrict__ dxx,
    DEVICE_REAL* __restrict__ kox) {
    const int g = IDX_PT(i, j, k, nx);
    const float sm3 = su[g - 3], sm2 = su[g - 2], sm1 = su[g - 1];
    const float s0  = su[g];
    const float sp1 = su[g + 1], sp2 = su[g + 2], sp3 = su[g + 3];
    const float inv = (float)inv_dx;
    if (dx)
        *dx = (DEVICE_REAL)((-sm3 + 9.0f * sm2 - 45.0f * sm1 + 45.0f * sp1 -
                             9.0f * sp2 + sp3) *
                            (inv / 60.0f));
    if (dxx)
        *dxx = (DEVICE_REAL)((2.0f * sm3 - 27.0f * sm2 + 270.0f * sm1 -
                              490.0f * s0 + 270.0f * sp1 - 27.0f * sp2 +
                              2.0f * sp3) *
                             ((inv * inv) / 180.0f));
    if (kox)
        *kox = (DEVICE_REAL)((-sm3 + 6.0f * sm2 - 15.0f * sm1 + 20.0f * s0 -
                              15.0f * sp1 + 6.0f * sp2 - sp3) *
                             (-inv / 64.0f));
}

template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_y3_f32su(
    const float* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dy,
    DEVICE_REAL* __restrict__ dy, DEVICE_REAL* __restrict__ dyy,
    DEVICE_REAL* __restrict__ koy) {
    const int g = IDX_PT(i, j, k, nx);
    const float sm3 = su[g - 3 * nx], sm2 = su[g - 2 * nx], sm1 = su[g - nx];
    const float s0  = su[g];
    const float sp1 = su[g + nx], sp2 = su[g + 2 * nx], sp3 = su[g + 3 * nx];
    const float inv = (float)inv_dy;
    if (dy)
        *dy = (DEVICE_REAL)((-sm3 + 9.0f * sm2 - 45.0f * sm1 + 45.0f * sp1 -
                             9.0f * sp2 + sp3) *
                            (inv / 60.0f));
    if (dyy)
        *dyy = (DEVICE_REAL)((2.0f * sm3 - 27.0f * sm2 + 270.0f * sm1 -
                              490.0f * s0 + 270.0f * sp1 - 27.0f * sp2 +
                              2.0f * sp3) *
                             ((inv * inv) / 180.0f));
    if (koy)
        *koy = (DEVICE_REAL)((-sm3 + 6.0f * sm2 - 15.0f * sm1 + 20.0f * s0 -
                              15.0f * sp1 + 6.0f * sp2 - sp3) *
                             (-inv / 64.0f));
}

template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_z3_f32su(
    const float* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dz,
    DEVICE_REAL* __restrict__ dz, DEVICE_REAL* __restrict__ dzz,
    DEVICE_REAL* __restrict__ koz) {
    const int nxy = nx * nx;
    const int g   = IDX_PT(i, j, k, nx);
    const float sm3 = su[g - 3 * nxy], sm2 = su[g - 2 * nxy], sm1 = su[g - nxy];
    const float s0  = su[g];
    const float sp1 = su[g + nxy], sp2 = su[g + 2 * nxy], sp3 = su[g + 3 * nxy];
    const float inv = (float)inv_dz;
    if (dz)
        *dz = (DEVICE_REAL)((-sm3 + 9.0f * sm2 - 45.0f * sm1 + 45.0f * sp1 -
                             9.0f * sp2 + sp3) *
                            (inv / 60.0f));
    if (dzz)
        *dzz = (DEVICE_REAL)((2.0f * sm3 - 27.0f * sm2 + 270.0f * sm1 -
                              490.0f * s0 + 270.0f * sp1 - 27.0f * sp2 +
                              2.0f * sp3) *
                             ((inv * inv) / 180.0f));
    if (koz)
        *koz = (DEVICE_REAL)((-sm3 + 6.0f * sm2 - 15.0f * sm1 + 20.0f * s0 -
                              15.0f * sp1 + 6.0f * sp2 - sp3) *
                             (-inv / 64.0f));
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dx_at_f32su(const float* __restrict__ su, int i, int j, int k,
                 DEVICE_REAL inv_dx) {
    const int g = IDX_PT(i, j, k, nx);
    const float inv = (float)inv_dx;
    return (DEVICE_REAL)((-su[g - 3] + 9.0f * su[g - 2] - 45.0f * su[g - 1] +
                          45.0f * su[g + 1] - 9.0f * su[g + 2] + su[g + 3]) *
                         (inv / 60.0f));
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dy_at_f32su(const float* __restrict__ su, int i, int j, int k,
                 DEVICE_REAL inv_dy) {
    const int g = IDX_PT(i, j, k, nx);
    const float inv = (float)inv_dy;
    return (DEVICE_REAL)((-su[g - 3 * nx] + 9.0f * su[g - 2 * nx] -
                          45.0f * su[g - nx] + 45.0f * su[g + nx] -
                          9.0f * su[g + 2 * nx] + su[g + 3 * nx]) *
                         (inv / 60.0f));
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dxy_f32su(const float* __restrict__ su, int i, int j, int k,
               DEVICE_REAL inv_dx, DEVICE_REAL inv_dy) {
    const DEVICE_REAL dxm3 = __pt_dx_at_f32su<nx>(su, i, j - 3, k, inv_dx);
    const DEVICE_REAL dxm2 = __pt_dx_at_f32su<nx>(su, i, j - 2, k, inv_dx);
    const DEVICE_REAL dxm1 = __pt_dx_at_f32su<nx>(su, i, j - 1, k, inv_dx);
    const DEVICE_REAL dxp1 = __pt_dx_at_f32su<nx>(su, i, j + 1, k, inv_dx);
    const DEVICE_REAL dxp2 = __pt_dx_at_f32su<nx>(su, i, j + 2, k, inv_dx);
    const DEVICE_REAL dxp3 = __pt_dx_at_f32su<nx>(su, i, j + 3, k, inv_dx);
    return (-dxm3 + 9.0 * dxm2 - 45.0 * dxm1 + 45.0 * dxp1 - 9.0 * dxp2 +
            dxp3) *
           (inv_dy / 60.0);
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dxz_f32su(const float* __restrict__ su, int i, int j, int k,
               DEVICE_REAL inv_dx, DEVICE_REAL inv_dz) {
    const DEVICE_REAL dxm3 = __pt_dx_at_f32su<nx>(su, i, j, k - 3, inv_dx);
    const DEVICE_REAL dxm2 = __pt_dx_at_f32su<nx>(su, i, j, k - 2, inv_dx);
    const DEVICE_REAL dxm1 = __pt_dx_at_f32su<nx>(su, i, j, k - 1, inv_dx);
    const DEVICE_REAL dxp1 = __pt_dx_at_f32su<nx>(su, i, j, k + 1, inv_dx);
    const DEVICE_REAL dxp2 = __pt_dx_at_f32su<nx>(su, i, j, k + 2, inv_dx);
    const DEVICE_REAL dxp3 = __pt_dx_at_f32su<nx>(su, i, j, k + 3, inv_dx);
    return (-dxm3 + 9.0 * dxm2 - 45.0 * dxm1 + 45.0 * dxp1 - 9.0 * dxp2 +
            dxp3) *
           (inv_dz / 60.0);
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dyz_f32su(const float* __restrict__ su, int i, int j, int k,
               DEVICE_REAL inv_dy, DEVICE_REAL inv_dz) {
    const DEVICE_REAL dym3 = __pt_dy_at_f32su<nx>(su, i, j, k - 3, inv_dy);
    const DEVICE_REAL dym2 = __pt_dy_at_f32su<nx>(su, i, j, k - 2, inv_dy);
    const DEVICE_REAL dym1 = __pt_dy_at_f32su<nx>(su, i, j, k - 1, inv_dy);
    const DEVICE_REAL dyp1 = __pt_dy_at_f32su<nx>(su, i, j, k + 1, inv_dy);
    const DEVICE_REAL dyp2 = __pt_dy_at_f32su<nx>(su, i, j, k + 2, inv_dy);
    const DEVICE_REAL dyp3 = __pt_dy_at_f32su<nx>(su, i, j, k + 3, inv_dy);
    return (-dym3 + 9.0 * dym2 - 45.0 * dym1 + 45.0 * dyp1 - 9.0 * dyp2 +
            dyp3) *
           (inv_dz / 60.0);
}

/** Same as __pt_fused_* but KO dissipation accumulated in FP32 (promoted back). */
template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_x3_kof32(
    const DEVICE_REAL* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dx,
    DEVICE_REAL* __restrict__ dx, DEVICE_REAL* __restrict__ dxx,
    DEVICE_REAL* __restrict__ kox) {
    const int g = IDX_PT(i, j, k, nx);
    const DEVICE_REAL sm3 = su[g - 3], sm2 = su[g - 2], sm1 = su[g - 1];
    const DEVICE_REAL s0  = su[g];
    const DEVICE_REAL sp1 = su[g + 1], sp2 = su[g + 2], sp3 = su[g + 3];
    if (dx)
        *dx = (-sm3 + 9.0 * sm2 - 45.0 * sm1 + 45.0 * sp1 - 9.0 * sp2 + sp3) *
              (inv_dx / 60.0);
    if (dxx)
        *dxx = (2.0 * sm3 - 27.0 * sm2 + 270.0 * sm1 - 490.0 * s0 + 270.0 * sp1 -
                27.0 * sp2 + 2.0 * sp3) *
               ((inv_dx * inv_dx) / 180.0);
    if (kox) {
        const float inv = (float)inv_dx;
        *kox = (DEVICE_REAL)((-(float)sm3 + 6.0f * (float)sm2 -
                              15.0f * (float)sm1 + 20.0f * (float)s0 -
                              15.0f * (float)sp1 + 6.0f * (float)sp2 -
                              (float)sp3) *
                             (-inv / 64.0f));
    }
}

template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_y3_kof32(
    const DEVICE_REAL* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dy,
    DEVICE_REAL* __restrict__ dy, DEVICE_REAL* __restrict__ dyy,
    DEVICE_REAL* __restrict__ koy) {
    const int g = IDX_PT(i, j, k, nx);
    const DEVICE_REAL sm3 = su[g - 3 * nx], sm2 = su[g - 2 * nx],
                      sm1 = su[g - nx];
    const DEVICE_REAL s0  = su[g];
    const DEVICE_REAL sp1 = su[g + nx], sp2 = su[g + 2 * nx],
                      sp3 = su[g + 3 * nx];
    if (dy)
        *dy = (-sm3 + 9.0 * sm2 - 45.0 * sm1 + 45.0 * sp1 - 9.0 * sp2 + sp3) *
              (inv_dy / 60.0);
    if (dyy)
        *dyy = (2.0 * sm3 - 27.0 * sm2 + 270.0 * sm1 - 490.0 * s0 + 270.0 * sp1 -
                27.0 * sp2 + 2.0 * sp3) *
               ((inv_dy * inv_dy) / 180.0);
    if (koy) {
        const float inv = (float)inv_dy;
        *koy = (DEVICE_REAL)((-(float)sm3 + 6.0f * (float)sm2 -
                              15.0f * (float)sm1 + 20.0f * (float)s0 -
                              15.0f * (float)sp1 + 6.0f * (float)sp2 -
                              (float)sp3) *
                             (-inv / 64.0f));
    }
}

template <int nx>
DEVICE_FUNC __forceinline__ void __pt_fused_z3_kof32(
    const DEVICE_REAL* __restrict__ su, int i, int j, int k, DEVICE_REAL inv_dz,
    DEVICE_REAL* __restrict__ dz, DEVICE_REAL* __restrict__ dzz,
    DEVICE_REAL* __restrict__ koz) {
    const int nxy = nx * nx;
    const int g   = IDX_PT(i, j, k, nx);
    const DEVICE_REAL sm3 = su[g - 3 * nxy], sm2 = su[g - 2 * nxy],
                      sm1 = su[g - nxy];
    const DEVICE_REAL s0  = su[g];
    const DEVICE_REAL sp1 = su[g + nxy], sp2 = su[g + 2 * nxy],
                      sp3 = su[g + 3 * nxy];
    if (dz)
        *dz = (-sm3 + 9.0 * sm2 - 45.0 * sm1 + 45.0 * sp1 - 9.0 * sp2 + sp3) *
              (inv_dz / 60.0);
    if (dzz)
        *dzz = (2.0 * sm3 - 27.0 * sm2 + 270.0 * sm1 - 490.0 * s0 + 270.0 * sp1 -
                27.0 * sp2 + 2.0 * sp3) *
               ((inv_dz * inv_dz) / 180.0);
    if (koz) {
        const float inv = (float)inv_dz;
        *koz = (DEVICE_REAL)((-(float)sm3 + 6.0f * (float)sm2 -
                              15.0f * (float)sm1 + 20.0f * (float)s0 -
                              15.0f * (float)sp1 + 6.0f * (float)sp2 -
                              (float)sp3) *
                             (-inv / 64.0f));
    }
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dx_at(const DEVICE_REAL* __restrict__ su, int i, int j, int k,
           DEVICE_REAL inv_dx) {
    const int g = IDX_PT(i, j, k, nx);
    return (-su[g - 3] + 9.0 * su[g - 2] - 45.0 * su[g - 1] + 45.0 * su[g + 1] -
            9.0 * su[g + 2] + su[g + 3]) *
           (inv_dx / 60.0);
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dy_at(const DEVICE_REAL* __restrict__ su, int i, int j, int k,
           DEVICE_REAL inv_dy) {
    const int g = IDX_PT(i, j, k, nx);
    return (-su[g - 3 * nx] + 9.0 * su[g - 2 * nx] - 45.0 * su[g - nx] +
            45.0 * su[g + nx] - 9.0 * su[g + 2 * nx] + su[g + 3 * nx]) *
           (inv_dy / 60.0);
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dxy(const DEVICE_REAL* __restrict__ su, int i, int j, int k,
         DEVICE_REAL inv_dx, DEVICE_REAL inv_dy) {
    const DEVICE_REAL dxm3 = __pt_dx_at<nx>(su, i, j - 3, k, inv_dx);
    const DEVICE_REAL dxm2 = __pt_dx_at<nx>(su, i, j - 2, k, inv_dx);
    const DEVICE_REAL dxm1 = __pt_dx_at<nx>(su, i, j - 1, k, inv_dx);
    const DEVICE_REAL dxp1 = __pt_dx_at<nx>(su, i, j + 1, k, inv_dx);
    const DEVICE_REAL dxp2 = __pt_dx_at<nx>(su, i, j + 2, k, inv_dx);
    const DEVICE_REAL dxp3 = __pt_dx_at<nx>(su, i, j + 3, k, inv_dx);
    return (-dxm3 + 9.0 * dxm2 - 45.0 * dxm1 + 45.0 * dxp1 - 9.0 * dxp2 +
            dxp3) *
           (inv_dy / 60.0);
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dxz(const DEVICE_REAL* __restrict__ su, int i, int j, int k,
         DEVICE_REAL inv_dx, DEVICE_REAL inv_dz) {
    const DEVICE_REAL dxm3 = __pt_dx_at<nx>(su, i, j, k - 3, inv_dx);
    const DEVICE_REAL dxm2 = __pt_dx_at<nx>(su, i, j, k - 2, inv_dx);
    const DEVICE_REAL dxm1 = __pt_dx_at<nx>(su, i, j, k - 1, inv_dx);
    const DEVICE_REAL dxp1 = __pt_dx_at<nx>(su, i, j, k + 1, inv_dx);
    const DEVICE_REAL dxp2 = __pt_dx_at<nx>(su, i, j, k + 2, inv_dx);
    const DEVICE_REAL dxp3 = __pt_dx_at<nx>(su, i, j, k + 3, inv_dx);
    return (-dxm3 + 9.0 * dxm2 - 45.0 * dxm1 + 45.0 * dxp1 - 9.0 * dxp2 +
            dxp3) *
           (inv_dz / 60.0);
}

template <int nx>
DEVICE_FUNC __forceinline__ DEVICE_REAL
__pt_dyz(const DEVICE_REAL* __restrict__ su, int i, int j, int k,
         DEVICE_REAL inv_dy, DEVICE_REAL inv_dz) {
    const DEVICE_REAL dym3 = __pt_dy_at<nx>(su, i, j, k - 3, inv_dy);
    const DEVICE_REAL dym2 = __pt_dy_at<nx>(su, i, j, k - 2, inv_dy);
    const DEVICE_REAL dym1 = __pt_dy_at<nx>(su, i, j, k - 1, inv_dy);
    const DEVICE_REAL dyp1 = __pt_dy_at<nx>(su, i, j, k + 1, inv_dy);
    const DEVICE_REAL dyp2 = __pt_dy_at<nx>(su, i, j, k + 2, inv_dy);
    const DEVICE_REAL dyp3 = __pt_dy_at<nx>(su, i, j, k + 3, inv_dy);
    return (-dym3 + 9.0 * dym2 - 45.0 * dym1 + 45.0 * dyp1 - 9.0 * dyp2 +
            dyp3) *
           (inv_dz / 60.0);
}
