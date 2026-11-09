/**
 * Interior-only FP32 smem stencils (bflag==0). Loop structure matches
 * dendrolib derivs_cu.cuh __blk1_deriv644_* / __blk1_ko_deriv42_*.
 */
#pragma once
// Include from inside namespace device { ... }

#ifndef IDX_B_F32
#define IDX_B_F32(i, j, k, nx) ((i) + (nx) * ((j) + (nx) * (k)))
#endif

template <int pw, unsigned int nx>
DEVICE_FUNC __inline__ void __ld_blk_var1_f32__(float* s_u,
                                                 const DEVICE_REAL* const var,
                                                 const BlockGPU3D* blk) {
    const DEVICE_INT tid =
        (GPUDevice::thread_id_z() * GPUDevice::block_dim_x() +
         GPUDevice::thread_id_y()) *
            GPUDevice::block_dim_y() +
        GPUDevice::thread_id_x();
    const DEVICE_INT n   = nx * nx * nx;
    const DEVICE_INT inc = GPUDevice::block_dim_x() * GPUDevice::block_dim_y() *
                           GPUDevice::block_dim_z();
    for (DEVICE_INT k = tid; k < n; k += inc) s_u[k] = (float)var[k];
    GPUDevice::sync_threads();
}

/** Cast (u − u_center) to float so quantization scales with block variation. */
template <int pw, unsigned int nx>
DEVICE_FUNC __inline__ void __ld_blk_var1_f32_centered__(
    float* s_u, const DEVICE_REAL* const var, const BlockGPU3D* blk) {
    const DEVICE_INT tid =
        (GPUDevice::thread_id_z() * GPUDevice::block_dim_x() +
         GPUDevice::thread_id_y()) *
            GPUDevice::block_dim_y() +
        GPUDevice::thread_id_x();
    const DEVICE_INT n   = nx * nx * nx;
    const DEVICE_INT inc = GPUDevice::block_dim_x() * GPUDevice::block_dim_y() *
                           GPUDevice::block_dim_z();
    // Interior center of the pw-padded block (matches thread (0,0,0) + pw).
    const DEVICE_INT c =
        pw + (DEVICE_INT)nx * (pw + (DEVICE_INT)nx * pw);
    // All threads reload var[c] from L1 instead of an extra __shared__ + sync.
    const DEVICE_REAL uc = var[c];
    for (DEVICE_INT k = tid; k < n; k += inc)
        s_u[k] = (float)(var[k] - uc);
    GPUDevice::sync_threads();
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_x_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float idx_by_60 = (1.0f / (float)blk->m_dx[0]) / 60.0f;
    const int nx = pencil_sz;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    for (int si = i + pw; si < actual - pw; si += inx)
        for (int sk = k; sk < actual; sk += inx)
            for (int sj = j; sj < actual; sj += inx) {
                const int g = IDX_B_F32(si, sj, sk, nx);
                Du[g] = (-s_u[g - 3] + 9.0f * s_u[g - 2] - 45.0f * s_u[g - 1] +
                         45.0f * s_u[g + 1] - 9.0f * s_u[g + 2] + s_u[g + 3]) *
                        idx_by_60;
            }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_y_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float idx_by_60 = (1.0f / (float)blk->m_dx[1]) / 60.0f;
    const int nx = pencil_sz;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    for (int si = i; si < actual; si += inx)
        for (int sk = k; sk < actual; sk += inx)
            for (int sj = j + pw; sj < actual - pw; sj += inx) {
                const int g = IDX_B_F32(si, sj, sk, nx);
                Du[g] =
                    (-s_u[g - 3 * nx] + 9.0f * s_u[g - 2 * nx] -
                     45.0f * s_u[g - nx] + 45.0f * s_u[g + nx] -
                     9.0f * s_u[g + 2 * nx] + s_u[g + 3 * nx]) *
                    idx_by_60;
            }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_z_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float idx_by_60 = (1.0f / (float)blk->m_dx[2]) / 60.0f;
    const int nx = pencil_sz;
    const int nxy = nx * nx;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    for (int si = i; si < actual; si += inx)
        for (int sj = j; sj < actual; sj += inx)
            for (int sk = k + pw; sk < actual - pw; sk += inx) {
                const int g = IDX_B_F32(si, sj, sk, nx);
                Du[g] =
                    (-s_u[g - 3 * nxy] + 9.0f * s_u[g - 2 * nxy] -
                     45.0f * s_u[g - nxy] + 45.0f * s_u[g + nxy] -
                     9.0f * s_u[g + 2 * nxy] + s_u[g + 3 * nxy]) *
                    idx_by_60;
            }
}

// Second derivs: dendrolib only evaluates the interior pencil for this thread's
// (j,k)/(i,k)/(i,j) — not a full nested volume sweep.
template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_xx_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float dx = (float)blk->m_dx[0];
    const float c = (1.0f / (dx * dx)) / 180.0f;
    const int nx = pencil_sz;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sk = k + pw;
    const int sj = j + pw;
    for (int si = i + pw; si < actual - pw; si += inx) {
        const int g = IDX_B_F32(si, sj, sk, nx);
        Du[g] = (2.0f * s_u[g - 3] - 27.0f * s_u[g - 2] + 270.0f * s_u[g - 1] -
                 490.0f * s_u[g] + 270.0f * s_u[g + 1] - 27.0f * s_u[g + 2] +
                 2.0f * s_u[g + 3]) *
                c;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_yy_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float dx = (float)blk->m_dx[1];
    const float c = (1.0f / (dx * dx)) / 180.0f;
    const int nx = pencil_sz;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sk = k + pw;
    const int si = i + pw;
    for (int sj = j + pw; sj < actual - pw; sj += inx) {
        const int g = IDX_B_F32(si, sj, sk, nx);
        Du[g] = (2.0f * s_u[g - 3 * nx] - 27.0f * s_u[g - 2 * nx] +
                 270.0f * s_u[g - nx] - 490.0f * s_u[g] + 270.0f * s_u[g + nx] -
                 27.0f * s_u[g + 2 * nx] + 2.0f * s_u[g + 3 * nx]) *
                c;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_deriv644_zz_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float dx = (float)blk->m_dx[2];
    const float c = (1.0f / (dx * dx)) / 180.0f;
    const int nx = pencil_sz;
    const int nxy = nx * nx;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sj = j + pw;
    const int si = i + pw;
    for (int sk = k + pw; sk < actual - pw; sk += inx) {
        const int g = IDX_B_F32(si, sj, sk, nx);
        Du[g] =
            (2.0f * s_u[g - 3 * nxy] - 27.0f * s_u[g - 2 * nxy] +
             270.0f * s_u[g - nxy] - 490.0f * s_u[g] + 270.0f * s_u[g + nxy] -
             27.0f * s_u[g + 2 * nxy] + 2.0f * s_u[g + 3 * nxy]) *
            c;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_ko_deriv42_x_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float dx = (float)blk->m_dx[0];
    const float pre = -1.0f / 64.0f / dx;
    const int nx = pencil_sz;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sk = k + pw;
    const int sj = j + pw;
    for (int si = i + pw; si < actual - pw; si += inx) {
        const int g = IDX_B_F32(si, sj, sk, nx);
        Du[g] = (-s_u[g - 3] + 6.0f * s_u[g - 2] - 15.0f * s_u[g - 1] +
                 20.0f * s_u[g] - 15.0f * s_u[g + 1] + 6.0f * s_u[g + 2] -
                 s_u[g + 3]) *
                pre;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_ko_deriv42_y_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float dx = (float)blk->m_dx[1];
    const float pre = -1.0f / 64.0f / dx;
    const int nx = pencil_sz;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sk = k + pw;
    const int si = i + pw;
    for (int sj = j + pw; sj < actual - pw; sj += inx) {
        const int g = IDX_B_F32(si, sj, sk, nx);
        Du[g] = (-s_u[g - 3 * nx] + 6.0f * s_u[g - 2 * nx] -
                 15.0f * s_u[g - nx] + 20.0f * s_u[g] - 15.0f * s_u[g + nx] +
                 6.0f * s_u[g + 2 * nx] - s_u[g + 3 * nx]) *
                pre;
    }
}

template <int pw, int pencil_sz>
DEVICE_FUNC __inline__ void __blk1_ko_deriv42_z_f32(
    float* __restrict__ const Du, const float* __restrict__ const s_u,
    const BlockGPU3D* const blk) {
    const float dx = (float)blk->m_dx[2];
    const float pre = -1.0f / 64.0f / dx;
    const int nx = pencil_sz;
    const int nxy = nx * nx;
    const int actual = 4 * pw + 1;
    const int i = GPUDevice::thread_id_x();
    const int j = GPUDevice::thread_id_y();
    const int k = GPUDevice::thread_id_z();
    const int inx = GPUDevice::block_dim_x();
    const int sj = j + pw;
    const int si = i + pw;
    for (int sk = k + pw; sk < actual - pw; sk += inx) {
        const int g = IDX_B_F32(si, sj, sk, nx);
        Du[g] = (-s_u[g - 3 * nxy] + 6.0f * s_u[g - 2 * nxy] -
                 15.0f * s_u[g - nxy] + 20.0f * s_u[g] - 15.0f * s_u[g + nxy] +
                 6.0f * s_u[g + 2 * nxy] - s_u[g + 3 * nxy]) *
                pre;
    }
}
