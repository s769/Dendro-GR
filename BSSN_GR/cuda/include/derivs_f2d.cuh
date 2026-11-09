/**
 * Float shared-field → FP64 pencil stencils (bflag==0).
 * Saves smem on su while keeping Du/DDu in double for mixed-deriv accuracy.
 */
#pragma once

#ifndef IDX_B_F2D
#define IDX_B_F2D(i, j, k, nx) ((i) + (nx) * ((j) + (nx) * (k)))
#endif

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_x_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL idx_by_60 = (1.0 / blk->m_dx[0]) / 60.0;
    const int nx = pencil_sz, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    for (int si = i + pw; si < actual - pw; si += inx)
        for (int sk = k; sk < actual; sk += inx)
            for (int sj = j; sj < actual; sj += inx) {
                const int g = IDX_B_F2D(si, sj, sk, nx);
                Du[g] = ((DEVICE_REAL)(-s_u[g - 3] + 9.f * s_u[g - 2] -
                                       45.f * s_u[g - 1] + 45.f * s_u[g + 1] -
                                       9.f * s_u[g + 2] + s_u[g + 3])) *
                        idx_by_60;
            }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_y_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL idx_by_60 = (1.0 / blk->m_dx[1]) / 60.0;
    const int nx = pencil_sz, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    for (int si = i; si < actual; si += inx)
        for (int sk = k; sk < actual; sk += inx)
            for (int sj = j + pw; sj < actual - pw; sj += inx) {
                const int g = IDX_B_F2D(si, sj, sk, nx);
                Du[g] =
                    ((DEVICE_REAL)(-s_u[g - 3 * nx] + 9.f * s_u[g - 2 * nx] -
                                   45.f * s_u[g - nx] + 45.f * s_u[g + nx] -
                                   9.f * s_u[g + 2 * nx] + s_u[g + 3 * nx])) *
                    idx_by_60;
            }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_z_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL idx_by_60 = (1.0 / blk->m_dx[2]) / 60.0;
    const int nx = pencil_sz, nxy = nx * nx, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    for (int si = i; si < actual; si += inx)
        for (int sj = j; sj < actual; sj += inx)
            for (int sk = k + pw; sk < actual - pw; sk += inx) {
                const int g = IDX_B_F2D(si, sj, sk, nx);
                Du[g] =
                    ((DEVICE_REAL)(-s_u[g - 3 * nxy] + 9.f * s_u[g - 2 * nxy] -
                                   45.f * s_u[g - nxy] + 45.f * s_u[g + nxy] -
                                   9.f * s_u[g + 2 * nxy] + s_u[g + 3 * nxy])) *
                    idx_by_60;
            }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_xx_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL dx = blk->m_dx[0];
    const DEVICE_REAL c = (1.0 / (dx * dx)) / 180.0;
    const int nx = pencil_sz, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sk = k + pw, sj = j + pw;
    for (int si = i + pw; si < actual - pw; si += inx) {
        const int g = IDX_B_F2D(si, sj, sk, nx);
        Du[g] = ((DEVICE_REAL)(2.f * s_u[g - 3] - 27.f * s_u[g - 2] +
                               270.f * s_u[g - 1] - 490.f * s_u[g] +
                               270.f * s_u[g + 1] - 27.f * s_u[g + 2] +
                               2.f * s_u[g + 3])) *
                c;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_yy_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL dx = blk->m_dx[1];
    const DEVICE_REAL c = (1.0 / (dx * dx)) / 180.0;
    const int nx = pencil_sz, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sk = k + pw, si = i + pw;
    for (int sj = j + pw; sj < actual - pw; sj += inx) {
        const int g = IDX_B_F2D(si, sj, sk, nx);
        Du[g] = ((DEVICE_REAL)(2.f * s_u[g - 3 * nx] - 27.f * s_u[g - 2 * nx] +
                               270.f * s_u[g - nx] - 490.f * s_u[g] +
                               270.f * s_u[g + nx] - 27.f * s_u[g + 2 * nx] +
                               2.f * s_u[g + 3 * nx])) *
                c;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_zz_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL dx = blk->m_dx[2];
    const DEVICE_REAL c = (1.0 / (dx * dx)) / 180.0;
    const int nx = pencil_sz, nxy = nx * nx, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sj = j + pw, si = i + pw;
    for (int sk = k + pw; sk < actual - pw; sk += inx) {
        const int g = IDX_B_F2D(si, sj, sk, nx);
        Du[g] =
            ((DEVICE_REAL)(2.f * s_u[g - 3 * nxy] - 27.f * s_u[g - 2 * nxy] +
                           270.f * s_u[g - nxy] - 490.f * s_u[g] +
                           270.f * s_u[g + nxy] - 27.f * s_u[g + 2 * nxy] +
                           2.f * s_u[g + 3 * nxy])) *
            c;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_ko_deriv42_x_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL dx = blk->m_dx[0];
    const DEVICE_REAL pre = -1.0 / 64.0 / dx;
    const int nx = pencil_sz, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sk = k + pw, sj = j + pw;
    for (int si = i + pw; si < actual - pw; si += inx) {
        const int g = IDX_B_F2D(si, sj, sk, nx);
        Du[g] = ((DEVICE_REAL)(-s_u[g - 3] + 6.f * s_u[g - 2] -
                               15.f * s_u[g - 1] + 20.f * s_u[g] -
                               15.f * s_u[g + 1] + 6.f * s_u[g + 2] -
                               s_u[g + 3])) *
                pre;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_ko_deriv42_y_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL dx = blk->m_dx[1];
    const DEVICE_REAL pre = -1.0 / 64.0 / dx;
    const int nx = pencil_sz, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sk = k + pw, si = i + pw;
    for (int sj = j + pw; sj < actual - pw; sj += inx) {
        const int g = IDX_B_F2D(si, sj, sk, nx);
        Du[g] = ((DEVICE_REAL)(-s_u[g - 3 * nx] + 6.f * s_u[g - 2 * nx] -
                               15.f * s_u[g - nx] + 20.f * s_u[g] -
                               15.f * s_u[g + nx] + 6.f * s_u[g + 2 * nx] -
                               s_u[g + 3 * nx])) *
                pre;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_ko_deriv42_z_f2d(
    DEVICE_REAL* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const DEVICE_REAL dx = blk->m_dx[2];
    const DEVICE_REAL pre = -1.0 / 64.0 / dx;
    const int nx = pencil_sz, nxy = nx * nx, actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sj = j + pw, si = i + pw;
    for (int sk = k + pw; sk < actual - pw; sk += inx) {
        const int g = IDX_B_F2D(si, sj, sk, nx);
        Du[g] = ((DEVICE_REAL)(-s_u[g - 3 * nxy] + 6.f * s_u[g - 2 * nxy] -
                               15.f * s_u[g - nxy] + 20.f * s_u[g] -
                               15.f * s_u[g + nxy] + 6.f * s_u[g + 2 * nxy] -
                               s_u[g + 3 * nxy])) *
                pre;
    }
}
