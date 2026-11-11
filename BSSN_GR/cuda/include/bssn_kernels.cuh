/**
 * @file bssn_deriv_kernels.cuh
 * @brief BSSN RHS derivative kernels.
 * @version 0.1
 * @date 2021-12-18
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once
// TODO: figure out proper ordering for these includes
// clang-format off
#include "device.h"
#include "block_gpu.h"
#include "grDef.h"
#include "bssnrhs_evar_derivs.cuh"
#include "derivs_cu.cuh"
#include "bc_cu.cuh"
#include <cuda_runtime_api.h> 
#include <cfloat>
#include <cstdlib>
#include <vector>
#include <cuda.h> 
#include <cooperative_groups.h>
// clang-format on

namespace device {
#include "derivs_f32.cuh"
#include "derivs_f2d.cuh"
#include "derivs_point.cuh"
/**
 * @brief Launch the kernel to compute all the x-direction derivatives
 */
template <int pw, int pencils, int pencil_sz, int BATCHED_BLOCKS_SZ>
GLOBAL_FUNC void launch_dir_x_deriv_kernel(const DEVICE_REAL* const u,
                                           BSSN_EVAR_DERIVS* deriv_evars,
                                           BlockGPU3D* blk, DEVICE_UINT nblocks,
                                           size_t szpdof_uz) {
    // includes x-derivs here.
    const DEVICE_UINT BLK_ID = GPUDevice::block_id_z();
    const DEVICE_UINT BLK_SZ = (blk[BLK_ID].m_aligned_sz[0]) *
                               (blk[BLK_ID].m_aligned_sz[1]) *
                               (blk[BLK_ID].m_aligned_sz[2]);
    const DEVICE_UINT offset     = blk[BLK_ID].m_offset;
    const size_t sz_per_dof = szpdof_uz;

    const DEVICE_REAL* const alpha =
        &u[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    const DEVICE_REAL* const chi = &u[bssn::VAR::U_CHI * sz_per_dof + offset];
    const DEVICE_REAL* const K   = &u[bssn::VAR::U_K * sz_per_dof + offset];
    const DEVICE_REAL* const gt0 =
        &u[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    const DEVICE_REAL* const gt1 =
        &u[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    const DEVICE_REAL* const gt2 =
        &u[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    const DEVICE_REAL* const gt3 =
        &u[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    const DEVICE_REAL* const gt4 =
        &u[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    const DEVICE_REAL* const gt5 =
        &u[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    const DEVICE_REAL* const beta0 =
        &u[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    const DEVICE_REAL* const beta1 =
        &u[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    const DEVICE_REAL* const beta2 =
        &u[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    const DEVICE_REAL* const At0 =
        &u[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    const DEVICE_REAL* const At1 =
        &u[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    const DEVICE_REAL* const At2 =
        &u[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    const DEVICE_REAL* const At3 =
        &u[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    const DEVICE_REAL* const At4 =
        &u[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    const DEVICE_REAL* const At5 =
        &u[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    const DEVICE_REAL* const Gt0 = &u[bssn::VAR::U_GT0 * sz_per_dof + offset];
    const DEVICE_REAL* const Gt1 = &u[bssn::VAR::U_GT1 * sz_per_dof + offset];
    const DEVICE_REAL* const Gt2 = &u[bssn::VAR::U_GT2 * sz_per_dof + offset];
    const DEVICE_REAL* const B0  = &u[bssn::VAR::U_B0 * sz_per_dof + offset];
    const DEVICE_REAL* const B1  = &u[bssn::VAR::U_B1 * sz_per_dof + offset];
    const DEVICE_REAL* const B2  = &u[bssn::VAR::U_B2 * sz_per_dof + offset];

#include "../scripts/bssnrhs_deriv_x_dir.cuh"

    return;
}

/**
 * @brief Launch the kernel to compute all the y-direction derivatives
 */
template <int pw, int pencils, int pencil_sz, int BATCHED_BLOCKS_SZ>
GLOBAL_FUNC void launch_dir_y_deriv_kernel(const DEVICE_REAL* const u,
                                           BSSN_EVAR_DERIVS* deriv_evars,
                                           BlockGPU3D* blk, DEVICE_UINT nblocks,
                                           size_t szpdof_uz) {
    // includes y-derivs here.
    const DEVICE_UINT BLK_ID = GPUDevice::block_id_z();
    const DEVICE_UINT BLK_SZ = (blk[BLK_ID].m_aligned_sz[0]) *
                               (blk[BLK_ID].m_aligned_sz[1]) *
                               (blk[BLK_ID].m_aligned_sz[2]);
    const DEVICE_UINT offset     = blk[BLK_ID].m_offset;
    const size_t sz_per_dof = szpdof_uz;

    const DEVICE_REAL* const alpha =
        &u[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    const DEVICE_REAL* const chi = &u[bssn::VAR::U_CHI * sz_per_dof + offset];
    const DEVICE_REAL* const K   = &u[bssn::VAR::U_K * sz_per_dof + offset];
    const DEVICE_REAL* const gt0 =
        &u[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    const DEVICE_REAL* const gt1 =
        &u[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    const DEVICE_REAL* const gt2 =
        &u[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    const DEVICE_REAL* const gt3 =
        &u[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    const DEVICE_REAL* const gt4 =
        &u[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    const DEVICE_REAL* const gt5 =
        &u[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    const DEVICE_REAL* const beta0 =
        &u[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    const DEVICE_REAL* const beta1 =
        &u[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    const DEVICE_REAL* const beta2 =
        &u[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    const DEVICE_REAL* const At0 =
        &u[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    const DEVICE_REAL* const At1 =
        &u[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    const DEVICE_REAL* const At2 =
        &u[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    const DEVICE_REAL* const At3 =
        &u[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    const DEVICE_REAL* const At4 =
        &u[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    const DEVICE_REAL* const At5 =
        &u[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    const DEVICE_REAL* const Gt0 = &u[bssn::VAR::U_GT0 * sz_per_dof + offset];
    const DEVICE_REAL* const Gt1 = &u[bssn::VAR::U_GT1 * sz_per_dof + offset];
    const DEVICE_REAL* const Gt2 = &u[bssn::VAR::U_GT2 * sz_per_dof + offset];
    const DEVICE_REAL* const B0  = &u[bssn::VAR::U_B0 * sz_per_dof + offset];
    const DEVICE_REAL* const B1  = &u[bssn::VAR::U_B1 * sz_per_dof + offset];
    const DEVICE_REAL* const B2  = &u[bssn::VAR::U_B2 * sz_per_dof + offset];

#include "../scripts/bssnrhs_deriv_y_dir.cuh"

    return;
}

template <typename T>
GLOBAL_FUNC __launch_bounds__(1024) void cuda_bssn_enforce_evar_cons(
    T* u, unsigned int lb, unsigned int le, T chi_floor, size_t szpdof,
    double* block_mins) {
    const T CHI_FLOOR = chi_floor;
    double t_min_det  = DBL_MAX;
    const T one_third = 1.0 / 3.0;
    T gtd[3][3], Atd[3][3];
    T* uiVar[bssn::BSSN_NUM_VARS];
    for (int i = 0; i < bssn::BSSN_NUM_VARS; i++) uiVar[i] = u + i * szpdof;

    const int t_id   = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int node = (t_id + lb); node < le; node += stride) {
        gtd[0][0] = uiVar[bssn::VAR::U_SYMGT0][node];
        gtd[0][1] = uiVar[bssn::VAR::U_SYMGT1][node];
        gtd[0][2] = uiVar[bssn::VAR::U_SYMGT2][node];
        gtd[1][0] = gtd[0][1];
        gtd[1][1] = uiVar[bssn::VAR::U_SYMGT3][node];
        gtd[1][2] = uiVar[bssn::VAR::U_SYMGT4][node];
        gtd[2][0] = gtd[0][2];
        gtd[2][1] = gtd[1][2];
        gtd[2][2] = uiVar[bssn::VAR::U_SYMGT5][node];

        Atd[0][0] = uiVar[bssn::VAR::U_SYMAT0][node];
        Atd[0][1] = uiVar[bssn::VAR::U_SYMAT1][node];
        Atd[0][2] = uiVar[bssn::VAR::U_SYMAT2][node];
        Atd[1][0] = Atd[0][1];
        Atd[1][1] = uiVar[bssn::VAR::U_SYMAT3][node];
        Atd[1][2] = uiVar[bssn::VAR::U_SYMAT4][node];
        Atd[2][0] = Atd[0][2];
        Atd[2][1] = Atd[1][2];
        Atd[2][2] = uiVar[bssn::VAR::U_SYMAT5][node];

        // printf("node %d\n gt00 = %.8E\n", node, gtd[0][0]);

        T det_gtd =
            gtd[0][0] * (gtd[1][1] * gtd[2][2] - gtd[1][2] * gtd[1][2]) -
            gtd[0][1] * gtd[0][1] * gtd[2][2] +
            2.0 * gtd[0][1] * gtd[0][2] * gtd[1][2] -
            gtd[0][2] * gtd[0][2] * gtd[1][1];
        // a NaN det is tracked as -inf
        const double det_trk = isnan(det_gtd) ? -INFINITY : det_gtd;
        if (det_trk < t_min_det) t_min_det = det_trk;

        if (det_gtd < 0.0) {
            printf("metric determinent = %.8E is negative \n", det_gtd);
            assert(false);
            // exit(0);
            /* FIXME What to do here? The metric is not physical. Do we reset
             * the metric to be flat? */
            gtd[0][0] = 1.0;
            gtd[0][1] = 0.0;
            gtd[0][2] = 0.0;
            gtd[1][0] = 0.0;
            gtd[1][1] = 1.0;
            gtd[1][2] = 0.0;
            gtd[2][0] = 0.0;
            gtd[2][1] = 0.0;
            gtd[2][2] = 1.0;
            det_gtd   = 1.0;
        }

        T det_gtd_to_neg_third = 1.0 / pow(det_gtd, one_third);

        for (unsigned int j = 0; j < 3; j++)
            for (unsigned int i = 0; i < 3; i++) {
                gtd[i][j] *= det_gtd_to_neg_third;
            }

        det_gtd = gtd[0][0] * (gtd[1][1] * gtd[2][2] - gtd[1][2] * gtd[1][2]) -
                  gtd[0][1] * gtd[0][1] * gtd[2][2] +
                  2.0 * gtd[0][1] * gtd[0][2] * gtd[1][2] -
                  gtd[0][2] * gtd[0][2] * gtd[1][1];

        T detgt_m1 = det_gtd - 1.0;

        if (fabs(detgt_m1) > 1.0e-6) {
            printf("enforce_bssn_constraint: det(gtd) != 1, det(gtd)=%.8E\n",
                   det_gtd);
            assert(false);
            // //std::cout.precision(14);
            // printf"enforce_bssn_constraint: det(gtd) != 1.
            // det="<<std::fixed<<det_gtd<<std::endl;
            //     std::cout<<"      gtd(1,1)="<<gtd[0][0]<<std::endl;
            //     std::cout<<"      gtd(1,2)="<<gtd[0][1]<<std::endl;
            //     std::cout<<"      gtd(1,3)="<<gtd[0][2]<<std::endl;
            //     std::cout<<"      gtd(2,2)="<<gtd[1][1]<<std::endl;
            //     std::cout<<"      gtd(2,3)="<<gtd[1][2]<<std::endl;
            //     std::cout<<"      gtd(3,3)="<<gtd[2][2]<<std::endl;

            // exit(0);
        }

        T gtu[3][3];
        T idet_gtd = 1.0 / det_gtd;
        gtu[0][0]  = idet_gtd * (gtd[1][1] * gtd[2][2] - gtd[1][2] * gtd[1][2]);
        gtu[0][1] = idet_gtd * (-gtd[0][1] * gtd[2][2] + gtd[0][2] * gtd[1][2]);
        gtu[0][2] = idet_gtd * (gtd[0][1] * gtd[1][2] - gtd[0][2] * gtd[1][1]);
        gtu[1][0] = gtu[0][1];
        gtu[1][1] = idet_gtd * (gtd[0][0] * gtd[2][2] - gtd[0][2] * gtd[0][2]);
        gtu[1][2] = idet_gtd * (-gtd[0][0] * gtd[1][2] + gtd[0][1] * gtd[0][2]);
        gtu[2][0] = gtu[0][2];
        gtu[2][1] = gtu[1][2];
        gtu[2][2] = idet_gtd * (gtd[0][0] * gtd[1][1] - gtd[0][1] * gtd[0][1]);

        /* Require Atd to be traceless. */
        T one_third_trace_Atd =
            one_third * (Atd[0][0] * gtu[0][0] + Atd[1][1] * gtu[1][1] +
                         Atd[2][2] * gtu[2][2] +
                         2.0 * (Atd[0][1] * gtu[0][1] + Atd[0][2] * gtu[0][2] +
                                Atd[1][2] * gtu[1][2]));

        Atd[0][0] -= one_third_trace_Atd * gtd[0][0];
        Atd[0][1] -= one_third_trace_Atd * gtd[0][1];
        Atd[0][2] -= one_third_trace_Atd * gtd[0][2];
        Atd[1][1] -= one_third_trace_Atd * gtd[1][1];
        Atd[1][2] -= one_third_trace_Atd * gtd[1][2];
        Atd[2][2] -= one_third_trace_Atd * gtd[2][2];

        T tr_A = Atd[0][0] * gtu[0][0] + Atd[1][1] * gtu[1][1] +
                 Atd[2][2] * gtu[2][2] +
                 2.0 * (Atd[0][1] * gtu[0][1] + Atd[0][2] * gtu[0][2] +
                        Atd[1][2] * gtu[1][2]);

        if (fabs(tr_A) > 1.0e-6) {
            printf("trace(A) != 0, trace(A)=%.8E\n", tr_A);
            assert(false);
            // std::cout<<"enforce_bssn_constraint: tr_A != 0.
            // tr_A="<<tr_A<<std::endl; std::cout<<"
            // Atd(1,1)="<<Atd[0][0]<<std::endl; std::cout<<"
            // Atd(1,2)="<<Atd[0][1]<<std::endl; std::cout<<"
            // Atd(1,3)="<<Atd[0][2]<<std::endl; std::cout<<"
            // Atd(2,2)="<<Atd[1][1]<<std::endl; std::cout<<"
            // Atd(2,3)="<<Atd[1][2]<<std::endl; std::cout<<"
            // Atd(3,3)="<<Atd[2][2]<<std::endl;

            // exit(0);
        }

        uiVar[bssn::VAR::U_SYMAT0][node] = Atd[0][0];
        uiVar[bssn::VAR::U_SYMAT1][node] = Atd[0][1];
        uiVar[bssn::VAR::U_SYMAT2][node] = Atd[0][2];
        uiVar[bssn::VAR::U_SYMAT3][node] = Atd[1][1];
        uiVar[bssn::VAR::U_SYMAT4][node] = Atd[1][2];
        uiVar[bssn::VAR::U_SYMAT5][node] = Atd[2][2];

        uiVar[bssn::VAR::U_SYMGT0][node] = gtd[0][0];
        uiVar[bssn::VAR::U_SYMGT1][node] = gtd[0][1];
        uiVar[bssn::VAR::U_SYMGT2][node] = gtd[0][2];
        uiVar[bssn::VAR::U_SYMGT3][node] = gtd[1][1];
        uiVar[bssn::VAR::U_SYMGT4][node] = gtd[1][2];
        uiVar[bssn::VAR::U_SYMGT5][node] = gtd[2][2];

        /* apply a floor to chi */
        if (uiVar[bssn::VAR::U_CHI][node] < CHI_FLOOR) {
            /* FIXME This needs to be fixed when we add a fluid to the code. */
            /* ! First rescale the densitized fluid variables.
                ! The include file bssn_puncture_fluid_rescale.inc
                ! must be provided in the BSSN_*MHD project.

                ! Chi must be positive to do the rescaling of fluid variables.
                if ( chi <= 0.0) {
                    chi = pars.chi_floor;
                }
                else {
                    // ok... go ahead and rescale the fluid variables.
                }


                */

            /* now place the floor on chi */
            uiVar[bssn::VAR::U_CHI][node] = CHI_FLOOR;
        }

        /* apply a floor to alpha */
        if (uiVar[bssn::VAR::U_ALPHA][node] < CHI_FLOOR)
            uiVar[bssn::VAR::U_ALPHA][node] = CHI_FLOOR;
    }

    // Order-fixed block reduction of pre-enforcement min det(gt).
    // Each CTA writes one value to block_mins[blockIdx.x]; the host then
    // folds block_mins[0..gridDim.x) in ascending block index
    // (see BSSNCtxGPU::post_timestep). No atomics — bit-reproducible.
    __shared__ double smin[32];  // one per warp (max 1024/32)
    const unsigned int lane   = threadIdx.x & 31;
    const unsigned int warp   = threadIdx.x >> 5;
    const unsigned int nwarps = (blockDim.x + 31) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        t_min_det =
            fmin(t_min_det, __shfl_down_sync(0xffffffff, t_min_det, offset));
    if (lane == 0) smin[warp] = t_min_det;
    __syncthreads();
    if (warp == 0) {
        double v = (lane < nwarps) ? smin[lane] : DBL_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmin(v, __shfl_down_sync(0xffffffff, v, offset));
        if (lane == 0 && block_mins) block_mins[blockIdx.x] = v;
    }
}

template <typename T, unsigned int pw, unsigned int nx>
DEVICE_FUNC __inline__ void __load_blk_var__(T* s_u, const T* const var,
                                             const BlockGPU3D* blk) {
    const DEVICE_INT i          = threadIdx.x;  // GPUDevice::thread_id_x();
    const DEVICE_INT j          = threadIdx.y;  // GPUDevice::thread_id_y();

    // const DEVICE_INT nx   = //blk->m_aligned_sz[0];
    // const DEVICE_INT ny   = //blk->m_aligned_sz[1];
    // const DEVICE_INT nz   = //blk->m_aligned_sz[2];

    const DEVICE_INT actual_nx  = 4 * pw + 1;  // blk->m_sz[0];
    const DEVICE_INT actual_ny  = 4 * pw + 1;  // blk->m_sz[1];
    const DEVICE_INT actual_nz  = 4 * pw + 1;  // blk->m_sz[2];

    const DEVICE_UINT globalIdx = j * nx + i;
    const DEVICE_INT ss         = nx * nx;

    if (i < actual_nx && j < actual_ny)
        for (DEVICE_INT k = 0; k < actual_nz; k++) {
            int idx  = globalIdx + k * ss;
            s_u[idx] = var[idx];
        }

    GPUDevice::sync_threads();
    return;
}

template <typename T, unsigned int pw, unsigned int nx>
DEVICE_FUNC __inline__ void __ld_blk_var1__(T* s_u, const T* const var,
                                            const BlockGPU3D* blk) {
    const DEVICE_INT i = GPUDevice::thread_id_x();
    const DEVICE_INT j = GPUDevice::thread_id_y();
    const DEVICE_INT k = GPUDevice::thread_id_z();

    const DEVICE_INT tid =
        (k * GPUDevice::block_dim_x() + j) * GPUDevice::block_dim_y() + i;

    const DEVICE_INT actual_nx = 4 * pw + 1;
    const DEVICE_INT actual_ny = 4 * pw + 1;
    const DEVICE_INT actual_nz = 4 * pw + 1;

    const DEVICE_INT n         = nx * nx * nx;
    const DEVICE_INT inc = GPUDevice::block_dim_x() * GPUDevice::block_dim_y() *
                           GPUDevice::block_dim_z();

    for (DEVICE_INT k = tid; k < n; k += inc) s_u[k] = var[k];

    GPUDevice::sync_threads();
    return;
}

template <typename T, unsigned int pw, unsigned int nx>
DEVICE_FUNC __inline__ void __st_blk_var1__(const T* const s_u,
                                            T* __restrict__ const var,
                                            const BlockGPU3D* blk) {
    const DEVICE_INT i = GPUDevice::thread_id_x();
    const DEVICE_INT j = GPUDevice::thread_id_y();
    const DEVICE_INT k = GPUDevice::thread_id_z();

    const DEVICE_INT tid =
        (k * GPUDevice::block_dim_x() + j) * GPUDevice::block_dim_y() + i;

    const DEVICE_INT actual_nx = 4 * pw + 1;
    const DEVICE_INT actual_ny = 4 * pw + 1;
    const DEVICE_INT actual_nz = 4 * pw + 1;

    const DEVICE_INT n         = nx * nx * nx;
    const DEVICE_INT inc = GPUDevice::block_dim_x() * GPUDevice::block_dim_y() *
                           GPUDevice::block_dim_z();

    for (DEVICE_INT k = tid; k < n; k += inc) var[k] = s_u[k];

    GPUDevice::sync_threads();
    return;
}

template <int pw, int nx, int pencil_sz>
DEVICE_FUNC void eval_rhs(DEVICE_REAL* const dptr_Fu,
                          const DEVICE_REAL* const dptr_u,
                          const BlockGPU3D* blk,
                          BSSN_EVAR_DERIVS* const deriv_evars,
                          size_t szpdof_uz) {
    const DEVICE_UINT sz_per_dof  = szpdof_uz;
    const DEVICE_UINT Z_ID        = GPUDevice::block_id_x();
    const DEVICE_UINT si          = GPUDevice::thread_id_x() + pw;
    const DEVICE_UINT sj          = GPUDevice::thread_id_y() + pw;

    const DEVICE_UINT lambda[4]   = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2] = {1.0, 0.0};
    const DEVICE_UINT offset      = blk->m_offset;
    const DEVICE_UINT actual_nx   = blk->m_sz[0];
    // const DEVICE_UINT nx              = blk[BLK_ID].m_aligned_sz[0];
    const DEVICE_UINT BLK_SZ      = nx * nx * nx;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_REAL hx              = blk->m_dx[0];
    const DEVICE_REAL hy              = blk->m_dx[1];
    const DEVICE_REAL hz              = blk->m_dx[2];
    const DEVICE_REAL x               = blk->m_ptMin[0] + si * hx;
    const DEVICE_REAL y               = blk->m_ptMin[1] + sj * hy;

    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;

    if (si < nx - pw && sj < nx - pw)
        for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
            const DEVICE_UINT pp      = sk * nx * nx + sj * nx + si;
            const DEVICE_REAL z       = blk->m_ptMin[2] + sk * hz;

            const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);

            const DEVICE_REAL w       = r_coord / RIT_ETA_WIDTH;
            const DEVICE_REAL arg     = -w * w * w * w;
            const DEVICE_REAL eta =
                (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

            const DEVICE_UINT d_pp = Z_ID * BLK_SZ + pp;
#include "../scripts/bssneqs_eta_const_standard_gauge4.cpp"
        }
}

template <int pw, int pencils, int pencil_sz, unsigned int BATCHED_GRAIN_SZ,
          unsigned int nx>
GLOBAL_FUNC __launch_bounds__(256) void bssnrhs(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
    BSSN_EVAR_DERIVS* const dptr_deriv_evars, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    const DEVICE_UINT NUM_BATCHES = nblocks / BATCHED_GRAIN_SZ + 1;
    const DEVICE_UINT blk_stride  = GPUDevice::grid_dim_x();
    const DEVICE_UINT Z_ID        = GPUDevice::block_id_x();
    const DEVICE_UINT block_begin = Z_ID;
    const DEVICE_UINT sz_per_dof  = szpdof_uz;

    SHARED_MEM DEVICE_REAL su[nx * nx * nx];
    BSSN_EVAR_DERIVS* const deriv_evars = dptr_deriv_evars;
    const DEVICE_UINT si                = GPUDevice::thread_id_x() + pw;
    const DEVICE_UINT sj                = GPUDevice::thread_id_y() + pw;

#pragma unroll
    for (unsigned int bid = block_begin; bid < nblocks; bid += blk_stride) {
        const BlockGPU3D current_blk = dptr_blk[bid];
        const BlockGPU3D* const blk  = &current_blk;
        // if(si ==pw && sj ==pw)
        //     printf("blk %d/%d with z_id %d\n",bid,nblocks,Z_ID);

        // const DEVICE_UINT BLK_ID            = bid;
        const DEVICE_UINT offset     = blk->m_offset;
        const DEVICE_UINT actual_nx  = blk->m_sz[0];
        // const DEVICE_UINT nx              = blk[BLK_ID].m_aligned_sz[0];
        const DEVICE_UINT BLK_SZ     = nx * nx * nx;
        {
            const DEVICE_REAL* const alpha =
                &dptr_u[bssn::VAR::U_ALPHA * sz_per_dof + offset];
            const DEVICE_REAL* const chi =
                &dptr_u[bssn::VAR::U_CHI * sz_per_dof + offset];
            const DEVICE_REAL* const K =
                &dptr_u[bssn::VAR::U_K * sz_per_dof + offset];
            const DEVICE_REAL* const gt0 =
                &dptr_u[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
            const DEVICE_REAL* const gt1 =
                &dptr_u[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
            const DEVICE_REAL* const gt2 =
                &dptr_u[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
            const DEVICE_REAL* const gt3 =
                &dptr_u[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
            const DEVICE_REAL* const gt4 =
                &dptr_u[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
            const DEVICE_REAL* const gt5 =
                &dptr_u[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
            const DEVICE_REAL* const beta0 =
                &dptr_u[bssn::VAR::U_BETA0 * sz_per_dof + offset];
            const DEVICE_REAL* const beta1 =
                &dptr_u[bssn::VAR::U_BETA1 * sz_per_dof + offset];
            const DEVICE_REAL* const beta2 =
                &dptr_u[bssn::VAR::U_BETA2 * sz_per_dof + offset];
            const DEVICE_REAL* const At0 =
                &dptr_u[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
            const DEVICE_REAL* const At1 =
                &dptr_u[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
            const DEVICE_REAL* const At2 =
                &dptr_u[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
            const DEVICE_REAL* const At3 =
                &dptr_u[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
            const DEVICE_REAL* const At4 =
                &dptr_u[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
            const DEVICE_REAL* const At5 =
                &dptr_u[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
            const DEVICE_REAL* const Gt0 =
                &dptr_u[bssn::VAR::U_GT0 * sz_per_dof + offset];
            const DEVICE_REAL* const Gt1 =
                &dptr_u[bssn::VAR::U_GT1 * sz_per_dof + offset];
            const DEVICE_REAL* const Gt2 =
                &dptr_u[bssn::VAR::U_GT2 * sz_per_dof + offset];
            const DEVICE_REAL* const B0 =
                &dptr_u[bssn::VAR::U_B0 * sz_per_dof + offset];
            const DEVICE_REAL* const B1 =
                &dptr_u[bssn::VAR::U_B1 * sz_per_dof + offset];
            const DEVICE_REAL* const B2 =
                &dptr_u[bssn::VAR::U_B2 * sz_per_dof + offset];

#include "../scripts/compute_bssnrhs_evar_derivs.cuh"
        }

        GPUDevice::sync_threads();

        // GPUDevice::sync_threads();
        //  if(si ==pw && sj ==pw)
        //      printf("blk %d/%d with z_id %d\n : derivs
        //      ended",bid,nblocks,Z_ID);

        eval_rhs<pw, nx, pencil_sz>(dptr_Fu, dptr_u, blk, deriv_evars,
                                    szpdof_uz);

        // GPUDevice::sync_threads();
        //  if(si ==pw && sj ==pw)
        //      printf("blk %d/%d with z_id %d\n : rhs ended",bid,nblocks,Z_ID);

        if (blk->m_bflag != 0) {
            DEVICE_REAL* const a_rhs =
                &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
            DEVICE_REAL* const chi_rhs =
                &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
            DEVICE_REAL* const gt_rhs00 =
                &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
            DEVICE_REAL* const gt_rhs01 =
                &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
            DEVICE_REAL* const gt_rhs02 =
                &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
            DEVICE_REAL* const gt_rhs11 =
                &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
            DEVICE_REAL* const gt_rhs12 =
                &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
            DEVICE_REAL* const gt_rhs22 =
                &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
            DEVICE_REAL* const b_rhs0 =
                &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
            DEVICE_REAL* const b_rhs1 =
                &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
            DEVICE_REAL* const b_rhs2 =
                &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
            DEVICE_REAL* const At_rhs00 =
                &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
            DEVICE_REAL* const At_rhs01 =
                &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
            DEVICE_REAL* const At_rhs02 =
                &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
            DEVICE_REAL* const At_rhs11 =
                &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
            DEVICE_REAL* const At_rhs12 =
                &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
            DEVICE_REAL* const At_rhs22 =
                &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
            DEVICE_REAL* const Gt_rhs0 =
                &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
            DEVICE_REAL* const Gt_rhs1 =
                &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
            DEVICE_REAL* const Gt_rhs2 =
                &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
            DEVICE_REAL* const B_rhs0 =
                &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
            DEVICE_REAL* const B_rhs1 =
                &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
            DEVICE_REAL* const B_rhs2 =
                &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
            DEVICE_REAL* const K_rhs =
                &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

            const DEVICE_REAL* const alpha =
                (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
            const DEVICE_REAL* const chi =
                (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
            const DEVICE_REAL* const K =
                (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
            const DEVICE_REAL* const gt0 =
                (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
            const DEVICE_REAL* const gt1 =
                (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
            const DEVICE_REAL* const gt2 =
                (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
            const DEVICE_REAL* const gt3 =
                (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
            const DEVICE_REAL* const gt4 =
                (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
            const DEVICE_REAL* const gt5 =
                (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
            const DEVICE_REAL* const beta0 =
                (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
            const DEVICE_REAL* const beta1 =
                (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
            const DEVICE_REAL* const beta2 =
                (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
            const DEVICE_REAL* const At0 =
                (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
            const DEVICE_REAL* const At1 =
                (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
            const DEVICE_REAL* const At2 =
                (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
            const DEVICE_REAL* const At3 =
                (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
            const DEVICE_REAL* const At4 =
                (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
            const DEVICE_REAL* const At5 =
                (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
            const DEVICE_REAL* const Gt0 =
                (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
            const DEVICE_REAL* const Gt1 =
                (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
            const DEVICE_REAL* const Gt2 =
                (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
            const DEVICE_REAL* const B0 =
                (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
            const DEVICE_REAL* const B1 =
                (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
            const DEVICE_REAL* const B2 =
                (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

            const DEVICE_REAL* const grad_0_alpha =
                (deriv_evars->grad_0_alpha + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_alpha =
                (deriv_evars->grad_1_alpha + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_alpha =
                (deriv_evars->grad_2_alpha + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_beta0 =
                (deriv_evars->grad_0_beta0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_beta0 =
                (deriv_evars->grad_1_beta0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_beta0 =
                (deriv_evars->grad_2_beta0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_beta1 =
                (deriv_evars->grad_0_beta1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_beta1 =
                (deriv_evars->grad_1_beta1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_beta1 =
                (deriv_evars->grad_2_beta1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_beta2 =
                (deriv_evars->grad_0_beta2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_beta2 =
                (deriv_evars->grad_1_beta2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_beta2 =
                (deriv_evars->grad_2_beta2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_B0 =
                (deriv_evars->grad_0_B0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_B0 =
                (deriv_evars->grad_1_B0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_B0 =
                (deriv_evars->grad_2_B0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_B1 =
                (deriv_evars->grad_0_B1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_B1 =
                (deriv_evars->grad_1_B1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_B1 =
                (deriv_evars->grad_2_B1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_B2 =
                (deriv_evars->grad_0_B2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_B2 =
                (deriv_evars->grad_1_B2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_B2 =
                (deriv_evars->grad_2_B2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_chi =
                (deriv_evars->grad_0_chi + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_chi =
                (deriv_evars->grad_1_chi + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_chi =
                (deriv_evars->grad_2_chi + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_Gt0 =
                (deriv_evars->grad_0_Gt0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_Gt0 =
                (deriv_evars->grad_1_Gt0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_Gt0 =
                (deriv_evars->grad_2_Gt0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_Gt1 =
                (deriv_evars->grad_0_Gt1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_Gt1 =
                (deriv_evars->grad_1_Gt1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_Gt1 =
                (deriv_evars->grad_2_Gt1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_Gt2 =
                (deriv_evars->grad_0_Gt2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_Gt2 =
                (deriv_evars->grad_1_Gt2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_Gt2 =
                (deriv_evars->grad_2_Gt2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_K =
                (deriv_evars->grad_0_K + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_K =
                (deriv_evars->grad_1_K + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_K =
                (deriv_evars->grad_2_K + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_gt0 =
                (deriv_evars->grad_0_gt0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_gt0 =
                (deriv_evars->grad_1_gt0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_gt0 =
                (deriv_evars->grad_2_gt0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_gt1 =
                (deriv_evars->grad_0_gt1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_gt1 =
                (deriv_evars->grad_1_gt1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_gt1 =
                (deriv_evars->grad_2_gt1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_gt2 =
                (deriv_evars->grad_0_gt2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_gt2 =
                (deriv_evars->grad_1_gt2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_gt2 =
                (deriv_evars->grad_2_gt2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_gt3 =
                (deriv_evars->grad_0_gt3 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_gt3 =
                (deriv_evars->grad_1_gt3 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_gt3 =
                (deriv_evars->grad_2_gt3 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_gt4 =
                (deriv_evars->grad_0_gt4 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_gt4 =
                (deriv_evars->grad_1_gt4 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_gt4 =
                (deriv_evars->grad_2_gt4 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_gt5 =
                (deriv_evars->grad_0_gt5 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_gt5 =
                (deriv_evars->grad_1_gt5 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_gt5 =
                (deriv_evars->grad_2_gt5 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_At0 =
                (deriv_evars->grad_0_At0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_At0 =
                (deriv_evars->grad_1_At0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_At0 =
                (deriv_evars->grad_2_At0 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_At1 =
                (deriv_evars->grad_0_At1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_At1 =
                (deriv_evars->grad_1_At1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_At1 =
                (deriv_evars->grad_2_At1 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_At2 =
                (deriv_evars->grad_0_At2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_At2 =
                (deriv_evars->grad_1_At2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_At2 =
                (deriv_evars->grad_2_At2 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_At3 =
                (deriv_evars->grad_0_At3 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_At3 =
                (deriv_evars->grad_1_At3 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_At3 =
                (deriv_evars->grad_2_At3 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_At4 =
                (deriv_evars->grad_0_At4 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_At4 =
                (deriv_evars->grad_1_At4 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_At4 =
                (deriv_evars->grad_2_At4 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_0_At5 =
                (deriv_evars->grad_0_At5 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_1_At5 =
                (deriv_evars->grad_1_At5 + Z_ID * BLK_SZ);
            const DEVICE_REAL* const grad_2_At5 =
                (deriv_evars->grad_2_At5 + Z_ID * BLK_SZ);

            radiative_bc_blk<pw>(a_rhs, alpha, grad_0_alpha, grad_1_alpha,
                                 grad_2_alpha, 1.0, 1.0, blk, 0);
            radiative_bc_blk<pw>(chi_rhs, chi, grad_0_chi, grad_1_chi,
                                 grad_2_chi, 1.0, 1.0, blk, 0);
            radiative_bc_blk<pw>(K_rhs, K, grad_0_K, grad_1_K, grad_2_K, 1.0,
                                 0.0, blk, 0);

            radiative_bc_blk<pw>(b_rhs0, beta0, grad_0_beta0, grad_1_beta0,
                                 grad_2_beta0, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(b_rhs1, beta1, grad_0_beta1, grad_1_beta1,
                                 grad_2_beta1, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(b_rhs2, beta2, grad_0_beta2, grad_1_beta2,
                                 grad_2_beta2, 1.0, 0.0, blk, 0);

            radiative_bc_blk<pw>(Gt_rhs0, Gt0, grad_0_Gt0, grad_1_Gt0,
                                 grad_2_Gt0, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(Gt_rhs1, Gt1, grad_0_Gt1, grad_1_Gt1,
                                 grad_2_Gt1, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(Gt_rhs2, Gt2, grad_0_Gt2, grad_1_Gt2,
                                 grad_2_Gt2, 2.0, 0.0, blk, 0);

            radiative_bc_blk<pw>(B_rhs0, B0, grad_0_B0, grad_1_B0, grad_2_B0,
                                 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(B_rhs1, B1, grad_0_B1, grad_1_B1, grad_2_B1,
                                 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(B_rhs2, B2, grad_0_B2, grad_1_B2, grad_2_B2,
                                 1.0, 0.0, blk, 0);

            radiative_bc_blk<pw>(At_rhs00, At0, grad_0_At0, grad_1_At0,
                                 grad_2_At0, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs01, At1, grad_0_At1, grad_1_At1,
                                 grad_2_At1, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs02, At2, grad_0_At2, grad_1_At2,
                                 grad_2_At2, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs11, At3, grad_0_At3, grad_1_At3,
                                 grad_2_At3, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs12, At4, grad_0_At4, grad_1_At4,
                                 grad_2_At4, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs22, At5, grad_0_At5, grad_1_At5,
                                 grad_2_At5, 2.0, 0.0, blk, 0);

            radiative_bc_blk<pw>(gt_rhs00, gt0, grad_0_gt0, grad_1_gt0,
                                 grad_2_gt0, 1.0, 1.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs01, gt1, grad_0_gt1, grad_1_gt1,
                                 grad_2_gt1, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs02, gt2, grad_0_gt2, grad_1_gt2,
                                 grad_2_gt2, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs11, gt3, grad_0_gt3, grad_1_gt3,
                                 grad_2_gt3, 1.0, 1.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs12, gt4, grad_0_gt4, grad_1_gt4,
                                 grad_2_gt4, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs22, gt5, grad_0_gt5, grad_1_gt5,
                                 grad_2_gt5, 1.0, 1.0, blk, 0);
        }

        const DEVICE_REAL sigma = ko_sigma;
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                DEVICE_REAL* const a_rhs =
                    &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
                DEVICE_REAL* const chi_rhs =
                    &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
                DEVICE_REAL* const gt_rhs00 =
                    &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
                DEVICE_REAL* const gt_rhs01 =
                    &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
                DEVICE_REAL* const gt_rhs02 =
                    &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
                DEVICE_REAL* const gt_rhs11 =
                    &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
                DEVICE_REAL* const gt_rhs12 =
                    &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
                DEVICE_REAL* const gt_rhs22 =
                    &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
                DEVICE_REAL* const b_rhs0 =
                    &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
                DEVICE_REAL* const b_rhs1 =
                    &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
                DEVICE_REAL* const b_rhs2 =
                    &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
                DEVICE_REAL* const At_rhs00 =
                    &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
                DEVICE_REAL* const At_rhs01 =
                    &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
                DEVICE_REAL* const At_rhs02 =
                    &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
                DEVICE_REAL* const At_rhs11 =
                    &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
                DEVICE_REAL* const At_rhs12 =
                    &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
                DEVICE_REAL* const At_rhs22 =
                    &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
                DEVICE_REAL* const Gt_rhs0 =
                    &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
                DEVICE_REAL* const Gt_rhs1 =
                    &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
                DEVICE_REAL* const Gt_rhs2 =
                    &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
                DEVICE_REAL* const B_rhs0 =
                    &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
                DEVICE_REAL* const B_rhs1 =
                    &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
                DEVICE_REAL* const B_rhs2 =
                    &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
                DEVICE_REAL* const K_rhs =
                    &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

                const DEVICE_UINT pp = sk * nx * nx + sj * nx + si;
#include "../scripts/bssnrhs_evar_derivs_ko_zid.h"
                a_rhs[pp] += sigma * (kograd_0_alpha[pp] + kograd_1_alpha[pp] +
                                      kograd_2_alpha[pp]);

                b_rhs0[pp] += sigma * (kograd_0_beta0[pp] + kograd_1_beta0[pp] +
                                       kograd_2_beta0[pp]);
                b_rhs1[pp] += sigma * (kograd_0_beta1[pp] + kograd_1_beta1[pp] +
                                       kograd_2_beta1[pp]);
                b_rhs2[pp] += sigma * (kograd_0_beta2[pp] + kograd_1_beta2[pp] +
                                       kograd_2_beta2[pp]);

                gt_rhs00[pp] += sigma * (kograd_0_gt0[pp] + kograd_1_gt0[pp] +
                                         kograd_2_gt0[pp]);
                gt_rhs01[pp] += sigma * (kograd_0_gt1[pp] + kograd_1_gt1[pp] +
                                         kograd_2_gt1[pp]);
                gt_rhs02[pp] += sigma * (kograd_0_gt2[pp] + kograd_1_gt2[pp] +
                                         kograd_2_gt2[pp]);
                gt_rhs11[pp] += sigma * (kograd_0_gt3[pp] + kograd_1_gt3[pp] +
                                         kograd_2_gt3[pp]);
                gt_rhs12[pp] += sigma * (kograd_0_gt4[pp] + kograd_1_gt4[pp] +
                                         kograd_2_gt4[pp]);
                gt_rhs22[pp] += sigma * (kograd_0_gt5[pp] + kograd_1_gt5[pp] +
                                         kograd_2_gt5[pp]);

                chi_rhs[pp] += sigma * (kograd_0_chi[pp] + kograd_1_chi[pp] +
                                        kograd_2_chi[pp]);

                At_rhs00[pp] += sigma * (kograd_0_At0[pp] + kograd_1_At0[pp] +
                                         kograd_2_At0[pp]);
                At_rhs01[pp] += sigma * (kograd_0_At1[pp] + kograd_1_At1[pp] +
                                         kograd_2_At1[pp]);
                At_rhs02[pp] += sigma * (kograd_0_At2[pp] + kograd_1_At2[pp] +
                                         kograd_2_At2[pp]);
                At_rhs11[pp] += sigma * (kograd_0_At3[pp] + kograd_1_At3[pp] +
                                         kograd_2_At3[pp]);
                At_rhs12[pp] += sigma * (kograd_0_At4[pp] + kograd_1_At4[pp] +
                                         kograd_2_At4[pp]);
                At_rhs22[pp] += sigma * (kograd_0_At5[pp] + kograd_1_At5[pp] +
                                         kograd_2_At5[pp]);

                K_rhs[pp] +=
                    sigma * (kograd_0_K[pp] + kograd_1_K[pp] + kograd_2_K[pp]);

                Gt_rhs0[pp] += sigma * (kograd_0_Gt0[pp] + kograd_1_Gt0[pp] +
                                        kograd_2_Gt0[pp]);
                Gt_rhs1[pp] += sigma * (kograd_0_Gt1[pp] + kograd_1_Gt1[pp] +
                                        kograd_2_Gt1[pp]);
                Gt_rhs2[pp] += sigma * (kograd_0_Gt2[pp] + kograd_1_Gt2[pp] +
                                        kograd_2_Gt2[pp]);

                B_rhs0[pp] += sigma * (kograd_0_B0[pp] + kograd_1_B0[pp] +
                                       kograd_2_B0[pp]);
                B_rhs1[pp] += sigma * (kograd_0_B1[pp] + kograd_1_B1[pp] +
                                       kograd_2_B1[pp]);
                B_rhs2[pp] += sigma * (kograd_0_B2[pp] + kograd_1_B2[pp] +
                                       kograd_2_B2[pp]);
            }

        // GPUDevice::sync_threads();
        //  if(si ==pw && sj ==pw)
        //      printf("blk %d/%d with z_id %d\n : ko  ended",bid,nblocks,Z_ID);
    }
}

template <int pw, unsigned int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(256, 2) void compute_all_derivs(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
    BSSN_EVAR_DERIVS* const dptr_deriv_evars, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    SHARED_MEM DEVICE_REAL su[(pencil_sz) * (pencil_sz) * (pencil_sz)];

    const DEVICE_INT Z_ID = GPUDevice::block_id_x();
    const DEVICE_INT V_ID = GPUDevice::block_id_y();

    if (Z_ID >= nblocks) return;

    const DEVICE_INT sz_per_dof                      = szpdof_uz;
    const int pencils                                = pencil_sz;

    BSSN_EVAR_DERIVS* __restrict__ const deriv_evars = dptr_deriv_evars;
    BlockGPU3D c_blk                                 = dptr_blk[Z_ID];

    const BlockGPU3D* __restrict__ const blk         = &c_blk;
    const DEVICE_UINT offset                         = blk->m_offset;
    const DEVICE_REAL* __restrict__ const var_in =
        &dptr_u[V_ID * sz_per_dof + offset];
    const DEVICE_UINT BLK_SZ = nx * nx * nx;

#include "../scripts/compute_bssnrhs_evar_derivs2.cuh"
}

// ---------------------------------------------------------------------------
// CAHD (constraint-adjusted Hamiltonian damping) -- see gen/gen_cahd.py.
// Pass 1 computes only the derivative arrays the CAHD bracket needs (chi, gt,
// Gt) into the batched workspace using the same per-variable derivative code
// as the stock two-kernel path (mixed derivatives included, one-sided stencils
// at physical boundaries); other members of BSSN_EVAR_DERIVS point at a dummy
// scratch buffer. Pass 2 evaluates the CPU's CAHD term pointwise.
// ---------------------------------------------------------------------------
template <int pw, unsigned int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(256, 2) void cahd_derivs(
    const DEVICE_REAL* const dptr_u, BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
    BSSN_EVAR_DERIVS* const dptr_deriv_evars, size_t szpdof_uz,
    DEVICE_UINT bflag_only) {
    SHARED_MEM DEVICE_REAL su[(pencil_sz) * (pencil_sz) * (pencil_sz)];
    const DEVICE_INT Z_ID = GPUDevice::block_id_x();
    const DEVICE_INT V_ID = GPUDevice::block_id_y();
    if (Z_ID >= nblocks) return;
    // VAR layout (grDef.h): chi = 1, Gt0..Gt2 = 3..5, SYMGT0..5 = 12..17
    if (!(V_ID == bssn::VAR::U_CHI ||
          (V_ID >= bssn::VAR::U_GT0 && V_ID <= bssn::VAR::U_GT2) ||
          (V_ID >= bssn::VAR::U_SYMGT0 && V_ID <= bssn::VAR::U_SYMGT5)))
        return;
    const DEVICE_INT sz_per_dof                      = szpdof_uz;
    const int pencils                                = pencil_sz;
    BSSN_EVAR_DERIVS* __restrict__ const deriv_evars = dptr_deriv_evars;
    BlockGPU3D c_blk                                 = dptr_blk[Z_ID];
    if (bflag_only && c_blk.m_bflag == 0) return;  // fused kernel handles these
    const BlockGPU3D* __restrict__ const blk         = &c_blk;
    const DEVICE_UINT offset                         = blk->m_offset;
    const DEVICE_REAL* __restrict__ const var_in =
        &dptr_u[V_ID * sz_per_dof + offset];
    const DEVICE_UINT BLK_SZ = nx * nx * nx;

#include "cahd_derivs_body.inc.cuh"
}

#include "cahd_gen.inc.cuh"

template <int pw, unsigned int nx>
GLOBAL_FUNC __launch_bounds__(343) void cahd_chi_rhs(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
    const BSSN_EVAR_DERIVS* const ws, size_t szpdof_uz, DEVICE_REAL cahd_C,
    DEVICE_REAL dt, DEVICE_UINT bflag_only) {
    const DEVICE_INT Z_ID = GPUDevice::block_id_x();
    if (Z_ID >= nblocks) return;
    if (bflag_only && dptr_blk[Z_ID].m_bflag == 0) return;
    constexpr unsigned int NI = nx - 2 * pw;  // interior points per direction
    const unsigned int tid    = GPUDevice::thread_id_x();
    if (tid >= NI * NI * NI) return;
    const unsigned int i = pw + tid % NI;
    const unsigned int j = pw + (tid / NI) % NI;
    const unsigned int k = pw + tid / (NI * NI);

    const BlockGPU3D blk       = dptr_blk[Z_ID];
    const DEVICE_UINT offset   = blk.m_offset;
    const DEVICE_UINT BLK_SZ   = nx * nx * nx;
    const size_t sz_per_dof = szpdof_uz;
    const DEVICE_UINT pp       = k * nx * nx + j * nx + i;

    // the CPU radiative boundary condition overwrites the RHS on these face
    // layers, so the CPU never applies the CAHD term there
    const DEVICE_UINT bf = blk.m_bflag;
    if (bf != 0) {
        if ((bf & (1u << OCT_DIR_LEFT)) && i == (unsigned)pw) return;
        if ((bf & (1u << OCT_DIR_RIGHT)) && i == nx - pw - 1) return;
        if ((bf & (1u << OCT_DIR_DOWN)) && j == (unsigned)pw) return;
        if ((bf & (1u << OCT_DIR_UP)) && j == nx - pw - 1) return;
        if ((bf & (1u << OCT_DIR_BACK)) && k == (unsigned)pw) return;
        if ((bf & (1u << OCT_DIR_FRONT)) && k == nx - pw - 1) return;
    }

#define CAHD_DECL_FIELD(NAME, VID) \
    const DEVICE_REAL* const NAME = &dptr_u[bssn::VAR::VID * sz_per_dof + offset];
    CAHD_FIELD_LIST(CAHD_DECL_FIELD)
#undef CAHD_DECL_FIELD
#define CAHD_DECL_DERIV(NAME) const DEVICE_REAL* const NAME = ws->NAME + Z_ID * BLK_SZ;
    CAHD_DERIV_LIST(CAHD_DECL_DERIV)
#undef CAHD_DECL_DERIV

    const DEVICE_REAL dx_i = fmin(fmin(blk.m_dx[0], blk.m_dx[1]), blk.m_dx[2]);
    const DEVICE_REAL DENDRO_18 = dx_i * dx_i;  // dx_i^2 (as in the CPU RHS)

    CAHD_BRACKET_BODY

    const DEVICE_REAL DENDRO_640 = (1.0 / 12.0) * chi[pp];
    dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset + pp] +=
        -cahd_C * DENDRO_18 * DENDRO_640 * cahd_bracket /
        (dt * (10 * DENDRO_18 + 1));
}

// Fused CAHD for blocks away from the physical boundary (bflag == 0): one CTA per
// block, one thread per interior point; all 72 derivative values live in
// registers (variable-by-variable through shared memory), no HBM workspace.
// Same stencils as the stock device derivative code (interior 6th order).
template <int pw, unsigned int nx>
GLOBAL_FUNC __launch_bounds__(343) void cahd_fused(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL cahd_C, DEVICE_REAL dt) {
    const DEVICE_INT Z_ID = GPUDevice::block_id_x();
    if (Z_ID >= nblocks) return;
    const BlockGPU3D blk = dptr_blk[Z_ID];
    if (blk.m_bflag != 0) return;  // handled by cahd_derivs + cahd_chi_rhs
    constexpr unsigned int NI = nx - 2 * pw;
    const unsigned int tid    = GPUDevice::thread_id_x();  // blockDim.x == NI^3
    const int i = pw + tid % NI;
    const int j = pw + (tid / NI) % NI;
    const int k = pw + tid / (NI * NI);
    const DEVICE_UINT offset     = blk.m_offset;
    const size_t sz_per_dof = szpdof_uz;
    const DEVICE_UINT pp         = k * nx * nx + j * nx + i;
    SHARED_MEM DEVICE_REAL s[nx * nx * nx];

    const DEVICE_REAL ix60   = (1.0 / blk.m_dx[0]) / 60.0;
    const DEVICE_REAL iy60   = (1.0 / blk.m_dx[1]) / 60.0;
    const DEVICE_REAL iz60   = (1.0 / blk.m_dx[2]) / 60.0;
    const DEVICE_REAL ixx180 = (1.0 / (blk.m_dx[0] * blk.m_dx[0])) / 180.0;
    const DEVICE_REAL iyy180 = (1.0 / (blk.m_dx[1] * blk.m_dx[1])) / 180.0;
    const DEVICE_REAL izz180 = (1.0 / (blk.m_dx[2] * blk.m_dx[2])) / 180.0;

#include "cahd_fused_derivs.inc.cuh"

#define CAHD_DECL_FIELD_S(NAME, VID) \
    const DEVICE_REAL NAME = dptr_u[(size_t)bssn::VAR::VID * sz_per_dof + offset + pp];
    CAHD_FIELD_LIST(CAHD_DECL_FIELD_S)
#undef CAHD_DECL_FIELD_S

    const DEVICE_REAL dx_i = fmin(fmin(blk.m_dx[0], blk.m_dx[1]), blk.m_dx[2]);
    const DEVICE_REAL DENDRO_18 = dx_i * dx_i;
#include "cahd_bracket_scalar.inc.cuh"
    const DEVICE_REAL DENDRO_640 = (1.0 / 12.0) * chi;
    dptr_Fu[(size_t)bssn::VAR::U_CHI * sz_per_dof + offset + pp] +=
        -cahd_C * DENDRO_18 * DENDRO_640 * cahd_bracket /
        (dt * (10 * DENDRO_18 + 1));
}

// Contracted-tensor CAHD (default, DENDRO_GPU_CAHD_MODE=2). Mathematically identical to the generated
// bracket (cahd_fused / cahd_chi_rhs), validated numerically (gen/ham_contracted.py: B = -12 H to 4e-13 on
// random inputs): the Ricci tensor is contracted with the inverse metric analytically, which needs
//   - the 6 metric components and their first derivatives (18),
//   - the operator  gi_lm d_l d_m  applied to each metric component (accumulated, never stored as 36 values),
//   - only 3 of the 9 d_i Gt_j (the divergence), read straight from global memory,
//   - chi with its derivatives, At, K.
// ~100 live values instead of 622 temporaries + 72 derivative values -> no register spills.
namespace cahd_c {
__device__ __forceinline__ constexpr int sym(int i, int j) {
    return (i == j) ? (i == 0 ? 0 : (i == 1 ? 3 : 5)) : ((i + j == 1) ? 1 : ((i + j == 2) ? 2 : 4));
}
}  // namespace cahd_c

template <int pw, unsigned int nx, int MINB = 1>
GLOBAL_FUNC __launch_bounds__(343, MINB) void cahd_fused_c(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL cahd_C, DEVICE_REAL dt) {
    using cahd_c::sym;
    constexpr int PB[6] = {0, 0, 0, 1, 1, 2};   // pair (b,c) of the packed symmetric index (compile-time)
    constexpr int PC[6] = {0, 1, 2, 1, 2, 2};
    const DEVICE_INT Z_ID = GPUDevice::block_id_x();
    if (Z_ID >= nblocks) return;
    const BlockGPU3D blk = dptr_blk[Z_ID];
    if (blk.m_bflag != 0) return;
    constexpr unsigned int NI = nx - 2 * pw;
    const unsigned int tid    = GPUDevice::thread_id_x();
    const int i = pw + tid % NI;
    const int j = pw + (tid / NI) % NI;
    const int k = pw + tid / (NI * NI);
    const DEVICE_UINT offset     = blk.m_offset;
    const size_t sz_per_dof = szpdof_uz;
    const DEVICE_UINT pp         = k * nx * nx + j * nx + i;
    SHARED_MEM DEVICE_REAL s[nx * nx * nx];
    const DEVICE_REAL* const U = dptr_u + offset;

    const DEVICE_REAL ix60   = (1.0 / blk.m_dx[0]) / 60.0;
    const DEVICE_REAL iy60   = (1.0 / blk.m_dx[1]) / 60.0;
    const DEVICE_REAL iz60   = (1.0 / blk.m_dx[2]) / 60.0;
    const DEVICE_REAL ixx180 = (1.0 / (blk.m_dx[0] * blk.m_dx[0])) / 180.0;
    const DEVICE_REAL iyy180 = (1.0 / (blk.m_dx[1] * blk.m_dx[1])) / 180.0;
    const DEVICE_REAL izz180 = (1.0 / (blk.m_dx[2] * blk.m_dx[2])) / 180.0;

    auto Sx  = [&](int di, int dj, int dk) { return s[(i + di) + nx * ((j + dj) + nx * (k + dk))]; };
    auto d1x = [&](int dj, int dk) { return (-Sx(-3,dj,dk)+9.0*Sx(-2,dj,dk)-45.0*Sx(-1,dj,dk)+45.0*Sx(1,dj,dk)-9.0*Sx(2,dj,dk)+Sx(3,dj,dk))*ix60; };
    auto d1y = [&](int di, int dk) { return (-Sx(di,-3,dk)+9.0*Sx(di,-2,dk)-45.0*Sx(di,-1,dk)+45.0*Sx(di,1,dk)-9.0*Sx(di,2,dk)+Sx(di,3,dk))*iy60; };
    auto d1z = [&](int di, int dj) { return (-Sx(di,dj,-3)+9.0*Sx(di,dj,-2)-45.0*Sx(di,dj,-1)+45.0*Sx(di,dj,1)-9.0*Sx(di,dj,2)+Sx(di,dj,3))*iz60; };
    auto st  = [&](const DEVICE_REAL* a, DEVICE_REAL c) { return (-a[0]+9.0*a[1]-45.0*a[2]+45.0*a[4]-9.0*a[5]+a[6])*c; };

    // fields at the point
    DEVICE_REAL g[6], At[6];
#pragma unroll
    for (int q = 0; q < 6; q++) {
        g[q]  = U[(size_t)(bssn::VAR::U_SYMGT0 + q) * sz_per_dof + pp];
        At[q] = U[(size_t)(bssn::VAR::U_SYMAT0 + q) * sz_per_dof + pp];
    }
    const DEVICE_REAL K   = U[(size_t)bssn::VAR::U_K * sz_per_dof + pp];
    const DEVICE_REAL chi = U[(size_t)bssn::VAR::U_CHI * sz_per_dof + pp];
    // inverse metric (adjugate / det, as the generator)
    const DEVICE_REAL det = g[0]*(g[3]*g[5]-g[4]*g[4]) - g[1]*(g[1]*g[5]-g[4]*g[2]) + g[2]*(g[1]*g[4]-g[3]*g[2]);
    DEVICE_REAL gi6[6];
    gi6[0] = (g[3]*g[5]-g[4]*g[4])/det; gi6[1] = (g[2]*g[4]-g[1]*g[5])/det; gi6[2] = (g[1]*g[4]-g[2]*g[3])/det;
    gi6[3] = (g[0]*g[5]-g[2]*g[2])/det; gi6[4] = (g[1]*g[2]-g[0]*g[4])/det; gi6[5] = (g[0]*g[3]-g[1]*g[1])/det;
    auto GI = [&](int a, int b) { return gi6[sym(a, b)]; };

    // metric components: first derivatives and  Tg[q] = gi_lm d_l d_m g_q
    DEVICE_REAL dg[3][6], Tg[6];
    for (int q = 0; q < 6; q++) {
        for (unsigned int w = tid; w < nx*nx*nx; w += blockDim.x) s[w] = U[(size_t)(bssn::VAR::U_SYMGT0 + q) * sz_per_dof + w];
        __syncthreads();
        const DEVICE_REAL c0 = Sx(0,0,0);
        dg[0][q] = d1x(0,0); dg[1][q] = d1y(0,0); dg[2][q] = d1z(0,0);
        const DEVICE_REAL xx = (2.0*Sx(-3,0,0)-27.0*Sx(-2,0,0)+270.0*Sx(-1,0,0)-490.0*c0+270.0*Sx(1,0,0)-27.0*Sx(2,0,0)+2.0*Sx(3,0,0))*ixx180;
        const DEVICE_REAL yy = (2.0*Sx(0,-3,0)-27.0*Sx(0,-2,0)+270.0*Sx(0,-1,0)-490.0*c0+270.0*Sx(0,1,0)-27.0*Sx(0,2,0)+2.0*Sx(0,3,0))*iyy180;
        const DEVICE_REAL zz = (2.0*Sx(0,0,-3)-27.0*Sx(0,0,-2)+270.0*Sx(0,0,-1)-490.0*c0+270.0*Sx(0,0,1)-27.0*Sx(0,0,2)+2.0*Sx(0,0,3))*izz180;
        DEVICE_REAL a[7];
#pragma unroll
        for (int r = -3; r <= 3; r++) a[r + 3] = d1x(r, 0);
        const DEVICE_REAL xy = st(a, iy60);
#pragma unroll
        for (int r = -3; r <= 3; r++) a[r + 3] = d1x(0, r);
        const DEVICE_REAL xz = st(a, iz60);
#pragma unroll
        for (int r = -3; r <= 3; r++) a[r + 3] = d1y(0, r);
        const DEVICE_REAL yz = st(a, iz60);
        Tg[q] = gi6[0]*xx + gi6[3]*yy + gi6[5]*zz + 2.0*(gi6[1]*xy + gi6[2]*xz + gi6[4]*yz);
        __syncthreads();
    }
    // chi: first derivatives, L = gi_lm d_l d_m chi
    DEVICE_REAL dchi[3], Lc;
    {
        for (unsigned int w = tid; w < nx*nx*nx; w += blockDim.x) s[w] = U[(size_t)bssn::VAR::U_CHI * sz_per_dof + w];
        __syncthreads();
        const DEVICE_REAL c0 = Sx(0,0,0);
        dchi[0] = d1x(0,0); dchi[1] = d1y(0,0); dchi[2] = d1z(0,0);
        const DEVICE_REAL xx = (2.0*Sx(-3,0,0)-27.0*Sx(-2,0,0)+270.0*Sx(-1,0,0)-490.0*c0+270.0*Sx(1,0,0)-27.0*Sx(2,0,0)+2.0*Sx(3,0,0))*ixx180;
        const DEVICE_REAL yy = (2.0*Sx(0,-3,0)-27.0*Sx(0,-2,0)+270.0*Sx(0,-1,0)-490.0*c0+270.0*Sx(0,1,0)-27.0*Sx(0,2,0)+2.0*Sx(0,3,0))*iyy180;
        const DEVICE_REAL zz = (2.0*Sx(0,0,-3)-27.0*Sx(0,0,-2)+270.0*Sx(0,0,-1)-490.0*c0+270.0*Sx(0,0,1)-27.0*Sx(0,0,2)+2.0*Sx(0,0,3))*izz180;
        DEVICE_REAL a[7];
#pragma unroll
        for (int r = -3; r <= 3; r++) a[r + 3] = d1x(r, 0);
        const DEVICE_REAL xy = st(a, iy60);
#pragma unroll
        for (int r = -3; r <= 3; r++) a[r + 3] = d1x(0, r);
        const DEVICE_REAL xz = st(a, iz60);
#pragma unroll
        for (int r = -3; r <= 3; r++) a[r + 3] = d1y(0, r);
        const DEVICE_REAL yz = st(a, iz60);
        Lc = gi6[0]*xx + gi6[3]*yy + gi6[5]*zz + 2.0*(gi6[1]*xy + gi6[2]*xz + gi6[4]*yz);
        __syncthreads();
    }
    // divergence of the evolved Gamma-tilde: d_x Gt0 + d_y Gt1 + d_z Gt2 (global memory, 6th order centered)
    DEVICE_REAL divG;
    {
        const DEVICE_REAL* G0 = U + (size_t)(bssn::VAR::U_GT0 + 0) * sz_per_dof + pp;
        const DEVICE_REAL* G1 = U + (size_t)(bssn::VAR::U_GT0 + 1) * sz_per_dof + pp;
        const DEVICE_REAL* G2 = U + (size_t)(bssn::VAR::U_GT0 + 2) * sz_per_dof + pp;
        divG = (-G0[-3]+9.0*G0[-2]-45.0*G0[-1]+45.0*G0[1]-9.0*G0[2]+G0[3])*ix60
             + (-G1[-3*(int)nx]+9.0*G1[-2*(int)nx]-45.0*G1[-(int)nx]+45.0*G1[(int)nx]-9.0*G1[2*(int)nx]+G1[3*(int)nx])*iy60
             + (-G2[-3*(int)(nx*nx)]+9.0*G2[-2*(int)(nx*nx)]-45.0*G2[-(int)(nx*nx)]+45.0*G2[(int)(nx*nx)]-9.0*G2[2*(int)(nx*nx)]+G2[3*(int)(nx*nx)])*iz60;
    }

    // Christoffel symbols of the conformal metric
    DEVICE_REAL C1[3][6], C2[3][6], CalGt[3], Lam[3];
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
        for (int q = 0; q < 6; q++) {
            const int b = PB[q], c = PC[q];
            C1[a][q] = 0.5 * (dg[c][sym(a, b)] + dg[b][sym(a, c)] - dg[a][sym(b, c)]);
        }
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
        for (int q = 0; q < 6; q++) C2[a][q] = GI(a,0)*C1[0][q] + GI(a,1)*C1[1][q] + GI(a,2)*C1[2][q];
#pragma unroll
    for (int a = 0; a < 3; a++) {
        DEVICE_REAL cg = 0.0;
#pragma unroll
        for (int b = 0; b < 3; b++)
#pragma unroll
            for (int c = 0; c < 3; c++) cg += GI(b, c) * C2[a][sym(b, c)];
        CalGt[a] = cg;
        DEVICE_REAL lm = 0.0;
#pragma unroll
        for (int b = 0; b < 3; b++)
#pragma unroll
            for (int c = 0; c < 3; c++) lm += GI(b, c) * dg[a][sym(b, c)];
        Lam[a] = lm;
    }
    DEVICE_REAL t3 = 0.0;
#pragma unroll
    for (int a = 0; a < 3; a++) t3 += CalGt[a] * 0.5 * Lam[a];
    DEVICE_REAL t4a = 0.0, t4b = 0.0;
#pragma unroll
    for (int a = 0; a < 3; a++)       // a == k
#pragma unroll
        for (int l = 0; l < 3; l++)
#pragma unroll
            for (int m = 0; m < 3; m++) {
                const DEVICE_REAL glm = GI(l, m);
                DEVICE_REAL x4a = 0.0, x4b = 0.0;
#pragma unroll
                for (int ii = 0; ii < 3; ii++) {
                    x4a += C2[a][sym(l, ii)] * C2[ii][sym(a, m)];
                    DEVICE_REAL E = GI(ii,0)*C1[a][sym(l,0)] + GI(ii,1)*C1[a][sym(l,1)] + GI(ii,2)*C1[a][sym(l,2)];  // E[k][l][i]
                    x4b += C2[a][sym(ii, m)] * E;
                }
                t4a += glm * x4a;
                t4b += glm * x4b;
            }
    t4a *= 2.0;
    DEVICE_REAL S1 = 0.0;
    {
        const DEVICE_REAL mult[6] = {1.0, 2.0, 2.0, 1.0, 2.0, 1.0};
#pragma unroll
        for (int q = 0; q < 6; q++) S1 += mult[q] * gi6[q] * Tg[q];
        S1 *= -0.5;
    }
    const DEVICE_REAL SRt = S1 + divG + t3 + t4a + t4b;
    DEVICE_REAL Q = 0.0, cgc = 0.0;
#pragma unroll
    for (int l = 0; l < 3; l++) {
#pragma unroll
        for (int m = 0; m < 3; m++) Q += GI(l, m) * dchi[l] * dchi[m];
        cgc += CalGt[l] * dchi[l];
    }
    const DEVICE_REAL SRphi = (2.0 / chi) * (Lc - cgc) - (5.0 / (2.0 * chi * chi)) * Q;
    // A_ij A^ij = tr( (gi At)^2 )
    DEVICE_REAL AA = 0.0;
#pragma unroll
    for (int a = 0; a < 3; a++)
#pragma unroll
        for (int b = 0; b < 3; b++) {
            DEVICE_REAL mab = 0.0, mba = 0.0;
#pragma unroll
            for (int c = 0; c < 3; c++) { mab += GI(a, c) * At[sym(c, b)]; mba += GI(b, c) * At[sym(c, a)]; }
            AA += mab * mba;
        }
    const DEVICE_REAL H = chi * (SRt + SRphi) - AA + (2.0 / 3.0) * K * K;
    const DEVICE_REAL dx_i = fmin(fmin(blk.m_dx[0], blk.m_dx[1]), blk.m_dx[2]);
    const DEVICE_REAL dx2  = dx_i * dx_i;
    dptr_Fu[(size_t)bssn::VAR::U_CHI * sz_per_dof + offset + pp] +=
        cahd_C * dx2 * chi * H / (dt * (10.0 * dx2 + 1.0));
}

template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(64) void eval_rhs2(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
    BSSN_EVAR_DERIVS* const deriv_evars, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    const size_t sz_per_dof = szpdof_uz;
    const DEVICE_UINT Z_ID       = GPUDevice::block_id_x();

    const DEVICE_UINT si         = GPUDevice::thread_id_x() + pw;
    const DEVICE_UINT sj         = GPUDevice::thread_id_y() + pw;

    if (Z_ID >= nblocks) return;

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;

    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    // const DEVICE_UINT nx              = blk[BLK_ID].m_aligned_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_REAL hx              = blk->m_dx[0];
    const DEVICE_REAL hy              = blk->m_dx[1];

    const DEVICE_REAL x               = blk->m_ptMin[0] + si * hx;
    const DEVICE_REAL y               = blk->m_ptMin[1] + sj * hy;
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;

    if (si < nx - pw && sj < nx - pw)
        for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
            const DEVICE_UINT pp      = sk * nx * nx + sj * nx + si;
            const DEVICE_REAL hz      = blk->m_dx[2];
            const DEVICE_REAL z       = blk->m_ptMin[2] + sk * hz;
            const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);

            const DEVICE_REAL w       = r_coord / RIT_ETA_WIDTH;
            const DEVICE_REAL arg     = -w * w * w * w;
            const DEVICE_REAL eta =
                (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

            const DEVICE_UINT d_pp = Z_ID * BLK_SZ + pp;
#include "../scripts/bssneqs_eta_const_standard_gauge4.cpp"
        }

    GPUDevice::sync_threads();
    if (blk->m_bflag != 0) {
#include "../scripts/bssnrhs_evar_derivs_zid.h"

        radiative_bc_blk<pw>(a_rhs, alpha, grad_0_alpha, grad_1_alpha,
                             grad_2_alpha, 1.0, 1.0, blk, 0);
        radiative_bc_blk<pw>(chi_rhs, chi, grad_0_chi, grad_1_chi, grad_2_chi,
                             1.0, 1.0, blk, 0);
        radiative_bc_blk<pw>(K_rhs, K, grad_0_K, grad_1_K, grad_2_K, 1.0, 0.0,
                             blk, 0);

        radiative_bc_blk<pw>(b_rhs0, beta0, grad_0_beta0, grad_1_beta0,
                             grad_2_beta0, 1.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(b_rhs1, beta1, grad_0_beta1, grad_1_beta1,
                             grad_2_beta1, 1.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(b_rhs2, beta2, grad_0_beta2, grad_1_beta2,
                             grad_2_beta2, 1.0, 0.0, blk, 0);

        radiative_bc_blk<pw>(Gt_rhs0, Gt0, grad_0_Gt0, grad_1_Gt0, grad_2_Gt0,
                             2.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(Gt_rhs1, Gt1, grad_0_Gt1, grad_1_Gt1, grad_2_Gt1,
                             2.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(Gt_rhs2, Gt2, grad_0_Gt2, grad_1_Gt2, grad_2_Gt2,
                             2.0, 0.0, blk, 0);

        radiative_bc_blk<pw>(B_rhs0, B0, grad_0_B0, grad_1_B0, grad_2_B0, 1.0,
                             0.0, blk, 0);
        radiative_bc_blk<pw>(B_rhs1, B1, grad_0_B1, grad_1_B1, grad_2_B1, 1.0,
                             0.0, blk, 0);
        radiative_bc_blk<pw>(B_rhs2, B2, grad_0_B2, grad_1_B2, grad_2_B2, 1.0,
                             0.0, blk, 0);

        radiative_bc_blk<pw>(At_rhs00, At0, grad_0_At0, grad_1_At0, grad_2_At0,
                             2.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(At_rhs01, At1, grad_0_At1, grad_1_At1, grad_2_At1,
                             2.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(At_rhs02, At2, grad_0_At2, grad_1_At2, grad_2_At2,
                             2.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(At_rhs11, At3, grad_0_At3, grad_1_At3, grad_2_At3,
                             2.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(At_rhs12, At4, grad_0_At4, grad_1_At4, grad_2_At4,
                             2.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(At_rhs22, At5, grad_0_At5, grad_1_At5, grad_2_At5,
                             2.0, 0.0, blk, 0);

        radiative_bc_blk<pw>(gt_rhs00, gt0, grad_0_gt0, grad_1_gt0, grad_2_gt0,
                             1.0, 1.0, blk, 0);
        radiative_bc_blk<pw>(gt_rhs01, gt1, grad_0_gt1, grad_1_gt1, grad_2_gt1,
                             1.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(gt_rhs02, gt2, grad_0_gt2, grad_1_gt2, grad_2_gt2,
                             1.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(gt_rhs11, gt3, grad_0_gt3, grad_1_gt3, grad_2_gt3,
                             1.0, 1.0, blk, 0);
        radiative_bc_blk<pw>(gt_rhs12, gt4, grad_0_gt4, grad_1_gt4, grad_2_gt4,
                             1.0, 0.0, blk, 0);
        radiative_bc_blk<pw>(gt_rhs22, gt5, grad_0_gt5, grad_1_gt5, grad_2_gt5,
                             1.0, 1.0, blk, 0);
    }

    GPUDevice::sync_threads();
    const DEVICE_REAL sigma = ko_sigma;
    if (si < nx - pw && sj < nx - pw)
        for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
            const DEVICE_UINT pp = sk * nx * nx + sj * nx + si;
#include "../scripts/bssnrhs_evar_derivs_zid.h"
            a_rhs[pp] += sigma * (kograd_0_alpha[pp] + kograd_1_alpha[pp] +
                                  kograd_2_alpha[pp]);

            b_rhs0[pp] += sigma * (kograd_0_beta0[pp] + kograd_1_beta0[pp] +
                                   kograd_2_beta0[pp]);
            b_rhs1[pp] += sigma * (kograd_0_beta1[pp] + kograd_1_beta1[pp] +
                                   kograd_2_beta1[pp]);
            b_rhs2[pp] += sigma * (kograd_0_beta2[pp] + kograd_1_beta2[pp] +
                                   kograd_2_beta2[pp]);

            gt_rhs00[pp] += sigma * (kograd_0_gt0[pp] + kograd_1_gt0[pp] +
                                     kograd_2_gt0[pp]);
            gt_rhs01[pp] += sigma * (kograd_0_gt1[pp] + kograd_1_gt1[pp] +
                                     kograd_2_gt1[pp]);
            gt_rhs02[pp] += sigma * (kograd_0_gt2[pp] + kograd_1_gt2[pp] +
                                     kograd_2_gt2[pp]);
            gt_rhs11[pp] += sigma * (kograd_0_gt3[pp] + kograd_1_gt3[pp] +
                                     kograd_2_gt3[pp]);
            gt_rhs12[pp] += sigma * (kograd_0_gt4[pp] + kograd_1_gt4[pp] +
                                     kograd_2_gt4[pp]);
            gt_rhs22[pp] += sigma * (kograd_0_gt5[pp] + kograd_1_gt5[pp] +
                                     kograd_2_gt5[pp]);

            chi_rhs[pp] += sigma * (kograd_0_chi[pp] + kograd_1_chi[pp] +
                                    kograd_2_chi[pp]);

            At_rhs00[pp] += sigma * (kograd_0_At0[pp] + kograd_1_At0[pp] +
                                     kograd_2_At0[pp]);
            At_rhs01[pp] += sigma * (kograd_0_At1[pp] + kograd_1_At1[pp] +
                                     kograd_2_At1[pp]);
            At_rhs02[pp] += sigma * (kograd_0_At2[pp] + kograd_1_At2[pp] +
                                     kograd_2_At2[pp]);
            At_rhs11[pp] += sigma * (kograd_0_At3[pp] + kograd_1_At3[pp] +
                                     kograd_2_At3[pp]);
            At_rhs12[pp] += sigma * (kograd_0_At4[pp] + kograd_1_At4[pp] +
                                     kograd_2_At4[pp]);
            At_rhs22[pp] += sigma * (kograd_0_At5[pp] + kograd_1_At5[pp] +
                                     kograd_2_At5[pp]);

            K_rhs[pp] +=
                sigma * (kograd_0_K[pp] + kograd_1_K[pp] + kograd_2_K[pp]);

            Gt_rhs0[pp] += sigma * (kograd_0_Gt0[pp] + kograd_1_Gt0[pp] +
                                    kograd_2_Gt0[pp]);
            Gt_rhs1[pp] += sigma * (kograd_0_Gt1[pp] + kograd_1_Gt1[pp] +
                                    kograd_2_Gt1[pp]);
            Gt_rhs2[pp] += sigma * (kograd_0_Gt2[pp] + kograd_1_Gt2[pp] +
                                    kograd_2_Gt2[pp]);

            B_rhs0[pp] +=
                sigma * (kograd_0_B0[pp] + kograd_1_B0[pp] + kograd_2_B0[pp]);
            B_rhs1[pp] +=
                sigma * (kograd_0_B1[pp] + kograd_1_B1[pp] + kograd_2_B1[pp]);
            B_rhs2[pp] +=
                sigma * (kograd_0_B2[pp] + kograd_1_B2[pp] + kograd_2_B2[pp]);
        }
}

template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(64) void eval_rhs2a(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
    BSSN_EVAR_DERIVS* const deriv_evars, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    const size_t sz_per_dof = szpdof_uz;
    const DEVICE_UINT Z_ID       = GPUDevice::block_id_x();
    const DEVICE_UINT V_ID       = GPUDevice::block_id_y();

    const DEVICE_UINT si         = GPUDevice::thread_id_x() + pw;
    const DEVICE_UINT sj         = GPUDevice::thread_id_y() + pw;

    if (Z_ID >= nblocks) return;

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;

    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    // const DEVICE_UINT nx              = blk[BLK_ID].m_aligned_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_REAL hx              = blk->m_dx[0];
    const DEVICE_REAL hy              = blk->m_dx[1];
    const DEVICE_REAL hz              = blk->m_dx[2];

    const DEVICE_REAL x               = blk->m_ptMin[0] + si * hx;
    const DEVICE_REAL y               = blk->m_ptMin[1] + sj * hy;

    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;

    if (V_ID == 0) {
        // printf("execute: vid %d",V_ID);
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                const DEVICE_UINT pp      = sk * nx * nx + sj * nx + si;
                const DEVICE_REAL z       = blk->m_ptMin[2] + sk * hz;
                const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);

                const DEVICE_REAL w       = r_coord / RIT_ETA_WIDTH;
                const DEVICE_REAL arg     = -w * w * w * w;
                const DEVICE_REAL eta =
                    (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) +
                    RIT_ETA_OUTER;
                const DEVICE_UINT d_pp = Z_ID * BLK_SZ + pp;

#include "../scripts/block_id_0.cpp"
            }

        GPUDevice::sync_threads();
        if (blk->m_bflag != 0) {
#include "../scripts/bssnrhs_evar_derivs_zid.h"
            radiative_bc_blk<pw>(a_rhs, alpha, grad_0_alpha, grad_1_alpha,
                                 grad_2_alpha, 1.0, 1.0, blk, 0);
            radiative_bc_blk<pw>(b_rhs0, beta0, grad_0_beta0, grad_1_beta0,
                                 grad_2_beta0, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(b_rhs1, beta1, grad_0_beta1, grad_1_beta1,
                                 grad_2_beta1, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(b_rhs2, beta2, grad_0_beta2, grad_1_beta2,
                                 grad_2_beta2, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(chi_rhs, chi, grad_0_chi, grad_1_chi,
                                 grad_2_chi, 1.0, 1.0, blk, 0);

            radiative_bc_blk<pw>(gt_rhs00, gt0, grad_0_gt0, grad_1_gt0,
                                 grad_2_gt0, 1.0, 1.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs01, gt1, grad_0_gt1, grad_1_gt1,
                                 grad_2_gt1, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs02, gt2, grad_0_gt2, grad_1_gt2,
                                 grad_2_gt2, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs11, gt3, grad_0_gt3, grad_1_gt3,
                                 grad_2_gt3, 1.0, 1.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs12, gt4, grad_0_gt4, grad_1_gt4,
                                 grad_2_gt4, 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(gt_rhs22, gt5, grad_0_gt5, grad_1_gt5,
                                 grad_2_gt5, 1.0, 1.0, blk, 0);
        }

        GPUDevice::sync_threads();
        const DEVICE_REAL sigma = ko_sigma;
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                const DEVICE_UINT pp = sk * nx * nx + sj * nx + si;
#include "../scripts/bssnrhs_evar_derivs_ko_zid.h"
                a_rhs[pp] += sigma * (kograd_0_alpha[pp] + kograd_1_alpha[pp] +
                                      kograd_2_alpha[pp]);

                b_rhs0[pp] += sigma * (kograd_0_beta0[pp] + kograd_1_beta0[pp] +
                                       kograd_2_beta0[pp]);
                b_rhs1[pp] += sigma * (kograd_0_beta1[pp] + kograd_1_beta1[pp] +
                                       kograd_2_beta1[pp]);
                b_rhs2[pp] += sigma * (kograd_0_beta2[pp] + kograd_1_beta2[pp] +
                                       kograd_2_beta2[pp]);

                gt_rhs00[pp] += sigma * (kograd_0_gt0[pp] + kograd_1_gt0[pp] +
                                         kograd_2_gt0[pp]);
                gt_rhs01[pp] += sigma * (kograd_0_gt1[pp] + kograd_1_gt1[pp] +
                                         kograd_2_gt1[pp]);
                gt_rhs02[pp] += sigma * (kograd_0_gt2[pp] + kograd_1_gt2[pp] +
                                         kograd_2_gt2[pp]);
                gt_rhs11[pp] += sigma * (kograd_0_gt3[pp] + kograd_1_gt3[pp] +
                                         kograd_2_gt3[pp]);
                gt_rhs12[pp] += sigma * (kograd_0_gt4[pp] + kograd_1_gt4[pp] +
                                         kograd_2_gt4[pp]);
                gt_rhs22[pp] += sigma * (kograd_0_gt5[pp] + kograd_1_gt5[pp] +
                                         kograd_2_gt5[pp]);

                chi_rhs[pp] += sigma * (kograd_0_chi[pp] + kograd_1_chi[pp] +
                                        kograd_2_chi[pp]);
            }

        return;
    }

    if (V_ID == 1) {
        // printf("execute: vid %d",V_ID);
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                const DEVICE_UINT pp      = sk * nx * nx + sj * nx + si;
                const DEVICE_REAL z       = blk->m_ptMin[2] + sk * hz;
                const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);

                const DEVICE_REAL w       = r_coord / RIT_ETA_WIDTH;
                const DEVICE_REAL arg     = -w * w * w * w;
                const DEVICE_REAL eta =
                    (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) +
                    RIT_ETA_OUTER;
                const DEVICE_UINT d_pp = Z_ID * BLK_SZ + pp;

#include "../scripts/block_id_1.cpp"
            }

        GPUDevice::sync_threads();
        if (blk->m_bflag != 0) {
#include "../scripts/bssnrhs_evar_derivs_zid.h"
            radiative_bc_blk<pw>(At_rhs00, At0, grad_0_At0, grad_1_At0,
                                 grad_2_At0, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs01, At1, grad_0_At1, grad_1_At1,
                                 grad_2_At1, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs02, At2, grad_0_At2, grad_1_At2,
                                 grad_2_At2, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs11, At3, grad_0_At3, grad_1_At3,
                                 grad_2_At3, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs12, At4, grad_0_At4, grad_1_At4,
                                 grad_2_At4, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(At_rhs22, At5, grad_0_At5, grad_1_At5,
                                 grad_2_At5, 2.0, 0.0, blk, 0);
        }

        GPUDevice::sync_threads();
        const DEVICE_REAL sigma = ko_sigma;
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                const DEVICE_UINT pp = sk * nx * nx + sj * nx + si;
#include "../scripts/bssnrhs_evar_derivs_ko_zid.h"
                At_rhs00[pp] += sigma * (kograd_0_At0[pp] + kograd_1_At0[pp] +
                                         kograd_2_At0[pp]);
                At_rhs01[pp] += sigma * (kograd_0_At1[pp] + kograd_1_At1[pp] +
                                         kograd_2_At1[pp]);
                At_rhs02[pp] += sigma * (kograd_0_At2[pp] + kograd_1_At2[pp] +
                                         kograd_2_At2[pp]);
                At_rhs11[pp] += sigma * (kograd_0_At3[pp] + kograd_1_At3[pp] +
                                         kograd_2_At3[pp]);
                At_rhs12[pp] += sigma * (kograd_0_At4[pp] + kograd_1_At4[pp] +
                                         kograd_2_At4[pp]);
                At_rhs22[pp] += sigma * (kograd_0_At5[pp] + kograd_1_At5[pp] +
                                         kograd_2_At5[pp]);
            }

        return;
    }

    if (V_ID == 2) {
        // printf("execute: vid %d",V_ID);
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                const DEVICE_UINT pp      = sk * nx * nx + sj * nx + si;
                const DEVICE_REAL z       = blk->m_ptMin[2] + sk * hz;
                const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);

                const DEVICE_REAL w       = r_coord / RIT_ETA_WIDTH;
                const DEVICE_REAL arg     = -w * w * w * w;
                const DEVICE_REAL eta =
                    (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) +
                    RIT_ETA_OUTER;
                const DEVICE_UINT d_pp = Z_ID * BLK_SZ + pp;

#include "../scripts/block_id_2.cpp"
            }

        GPUDevice::sync_threads();
        if (blk->m_bflag != 0) {
#include "../scripts/bssnrhs_evar_derivs_zid.h"
            radiative_bc_blk<pw>(Gt_rhs0, Gt0, grad_0_Gt0, grad_1_Gt0,
                                 grad_2_Gt0, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(Gt_rhs1, Gt1, grad_0_Gt1, grad_1_Gt1,
                                 grad_2_Gt1, 2.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(Gt_rhs2, Gt2, grad_0_Gt2, grad_1_Gt2,
                                 grad_2_Gt2, 2.0, 0.0, blk, 0);

            radiative_bc_blk<pw>(B_rhs0, B0, grad_0_B0, grad_1_B0, grad_2_B0,
                                 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(B_rhs1, B1, grad_0_B1, grad_1_B1, grad_2_B1,
                                 1.0, 0.0, blk, 0);
            radiative_bc_blk<pw>(B_rhs2, B2, grad_0_B2, grad_1_B2, grad_2_B2,
                                 1.0, 0.0, blk, 0);
        }

        GPUDevice::sync_threads();
        const DEVICE_REAL sigma = ko_sigma;
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                const DEVICE_UINT pp = sk * nx * nx + sj * nx + si;
#include "../scripts/bssnrhs_evar_derivs_ko_zid.h"

                Gt_rhs0[pp] += sigma * (kograd_0_Gt0[pp] + kograd_1_Gt0[pp] +
                                        kograd_2_Gt0[pp]);
                Gt_rhs1[pp] += sigma * (kograd_0_Gt1[pp] + kograd_1_Gt1[pp] +
                                        kograd_2_Gt1[pp]);
                Gt_rhs2[pp] += sigma * (kograd_0_Gt2[pp] + kograd_1_Gt2[pp] +
                                        kograd_2_Gt2[pp]);

                B_rhs0[pp] += sigma * (kograd_0_B0[pp] + kograd_1_B0[pp] +
                                       kograd_2_B0[pp]);
                B_rhs1[pp] += sigma * (kograd_0_B1[pp] + kograd_1_B1[pp] +
                                       kograd_2_B1[pp]);
                B_rhs2[pp] += sigma * (kograd_0_B2[pp] + kograd_1_B2[pp] +
                                       kograd_2_B2[pp]);
            }

        return;
    }

    if (V_ID == 3) {
        // printf("execute: vid %d",V_ID);
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                const DEVICE_UINT pp      = sk * nx * nx + sj * nx + si;
                const DEVICE_REAL z       = blk->m_ptMin[2] + sk * hz;
                const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);

                const DEVICE_REAL w       = r_coord / RIT_ETA_WIDTH;
                const DEVICE_REAL arg     = -w * w * w * w;
                const DEVICE_REAL eta =
                    (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) +
                    RIT_ETA_OUTER;
                const DEVICE_UINT d_pp = Z_ID * BLK_SZ + pp;

#include "../scripts/block_id_3.cpp"
            }

        GPUDevice::sync_threads();
        if (blk->m_bflag != 0) {
#include "../scripts/bssnrhs_evar_derivs_zid.h"
            radiative_bc_blk<pw>(K_rhs, K, grad_0_K, grad_1_K, grad_2_K, 1.0,
                                 0.0, blk, 0);
        }

        GPUDevice::sync_threads();
        const DEVICE_REAL sigma = ko_sigma;
        if (si < nx - pw && sj < nx - pw)
            for (DEVICE_UINT sk = pw; sk < actual_nx - pw; sk++) {
                const DEVICE_UINT pp = sk * nx * nx + sj * nx + si;
#include "../scripts/bssnrhs_evar_derivs_ko_zid.h"

                K_rhs[pp] +=
                    sigma * (kograd_0_K[pp] + kograd_1_K[pp] + kograd_2_K[pp]);
            }

        return;
    }

    return;
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs2(DEVICE_REAL* const dptr_Fu,
                        const DEVICE_REAL* const dptr_u, BlockGPU3D* dptr_blk,
                        DEVICE_UINT nblocks,
                        BSSN_EVAR_DERIVS* const dptr_deriv_evars,
                        size_t szpdof_uz, DEVICE_REAL ko_sigma) {
    dim3 gb  = dim3(BATCHED_GRAIN_SZ, bssn::BSSN_NUM_VARS, 1);
    // dim3 tb = dim3(pencil_sz, pencil_sz, 1);
    dim3 tb  = dim3(13, 13, 1);
    dim3 gb2 = dim3(BATCHED_GRAIN_SZ, 1, 1);
    // dim3 gb2a = dim3(BATCHED_GRAIN_SZ, 4, 1);
    dim3 tb2 = dim3(8, 8, 1);
    cudaStream_t stream[nstreams];
    for (unsigned int i = 0; i < nstreams; i++) cudaStreamCreate(&stream[i]);

    const unsigned int num_batches = nblocks / BATCHED_GRAIN_SZ + 1;
    for (unsigned int batch_id = 0; batch_id < num_batches; batch_id++) {
        const unsigned int blk_begin = (batch_id * nblocks) / num_batches;
        const unsigned int blk_end   = ((batch_id + 1) * nblocks) / num_batches;
        const unsigned int sid       = batch_id % nstreams;
        const unsigned int nblock_p_batch = blk_end - blk_begin;
        compute_all_derivs<pw, nx, pencil_sz><<<gb, tb, 0, stream[sid]>>>(
            dptr_Fu, dptr_u, dptr_blk + blk_begin, nblock_p_batch,
            dptr_deriv_evars + sid, szpdof_uz, ko_sigma);
        eval_rhs2<pw, nx, pencil_sz><<<gb2, tb2, 0, stream[sid]>>>(
            dptr_Fu, dptr_u, dptr_blk + blk_begin, nblock_p_batch,
            dptr_deriv_evars + sid, szpdof_uz, ko_sigma);
        // eval_rhs2a<pw,nx,pencil_sz><<< gb2a,tb2, 0, stream[sid] >>>(dptr_Fu,
        // dptr_u, dptr_blk + blk_begin, nblock_p_batch, dptr_deriv_evars +sid,
        // szpdof_uz,ko_sigma);
    }

    for (unsigned int i = 0; i < nstreams; i++) cudaStreamDestroy(stream[i]);

    cudaDeviceSynchronize();

    return;
}

// bflag_sel: 0 = all blocks, 1 = interior only (m_bflag==0),
//            2 = domain-boundary only (m_bflag!=0). Used by hybrid PT launch.
template <int pw, int nx, int pencil_sz, int bflag_sel = 0>
#ifndef BSSN_STOCK_MINB
#define BSSN_STOCK_MINB 3
#endif
GLOBAL_FUNC __launch_bounds__(343, BSSN_STOCK_MINB) void eval_rhs3(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    extern __shared__ DEVICE_REAL shared_mem[];
    // __shared__ DEVICE_REAL su[pencil_sz * pencil_sz *pencil_sz];
    // __shared__ DEVICE_REAL Du[pencil_sz * pencil_sz *pencil_sz];

    DEVICE_REAL* su  = shared_mem;
    DEVICE_REAL* Du  = shared_mem + pencil_sz * pencil_sz * pencil_sz;
    DEVICE_REAL* DDu = shared_mem + 2 * pencil_sz * pencil_sz * pencil_sz;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    if constexpr (bflag_sel == 1) {
        if (blk->m_bflag != 0) return;
    } else if constexpr (bflag_sel == 2) {
        if (blk->m_bflag == 0) return;
    }

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i                = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j                = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k                = GPUDevice::thread_id_z() + pw;

    const int gidx                    = (k * nx + j) * nx + i;

    // #include "../scripts/compute_bssnrhs_evar_derivs3.cuh"
    const DEVICE_INT pp               = gidx;

    const DEVICE_REAL hx              = blk->m_dx[0];

    const DEVICE_REAL x               = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y               = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z               = blk->m_ptMin[2] + k * blk->m_dx[2];

    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;

    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);

    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

// #pragma message ("key_expr_sympy_cse_with_fused")
#include "../scripts/bssneqs_eta_const_standard_gauge6.cpp"

    /*#include "../scripts/compute_bssnrhs_evar_derivs3.cuh"
    //#include "../scripts/bssneqs_eta_const_standard_gauge5.cpp"

    // #pragma message("sympy_cse")
    // #include "../../../CodeGen/bssneqs_sympy_cse_wo_derivs.cpp"

    // #pragma message("nx_cse_without_derivs")
    // #include "../../../CodeGen/bssneqs_nx_cse_wo_derivs.cpp"

    // #pragma message("key_exprs")
    // #include "../../../CodeGen/bssneqs_key_expr_sympy_cse_wo_derivs.cpp"



    GPUDevice::sync_threads();
    if(blk->m_bflag != 0)
    {
        radiative_bc_pt<pw , nx>(&a_rhs [gidx] , alpha[gidx], grad_0_alpha,
    grad_1_alpha, grad_2_alpha,1.0, 1.0,blk);

        radiative_bc_pt<pw , nx>(&b_rhs0[gidx] , beta0[gidx], grad_0_beta0,
    grad_1_beta0, grad_2_beta0, 1.0, 0.0, blk); radiative_bc_pt<pw ,
    nx>(&b_rhs1[gidx] , beta1[gidx], grad_0_beta1, grad_1_beta1,
    grad_2_beta1, 1.0, 0.0, blk); radiative_bc_pt<pw , nx>(&b_rhs2[gidx] ,
    beta2[gidx], grad_0_beta2, grad_1_beta2, grad_2_beta2, 1.0, 0.0, blk);

        radiative_bc_pt<pw , nx>(&chi_rhs[gidx], chi[gidx]  , grad_0_chi,
    grad_1_chi, grad_2_chi, 1.0, 1.0, blk); radiative_bc_pt<pw ,
    nx>(&K_rhs[gidx],     K[gidx]  ,   grad_0_K,   grad_1_K,   grad_2_K, 1.0,
    0.0, blk);

        radiative_bc_pt<pw , nx>(&Gt_rhs0[gidx], Gt0[gidx], grad_0_Gt0,
    grad_1_Gt0, grad_2_Gt0, 2.0, 0.0, blk); radiative_bc_pt<pw ,
    nx>(&Gt_rhs1[gidx], Gt1[gidx], grad_0_Gt1, grad_1_Gt1, grad_2_Gt1, 2.0, 0.0,
    blk); radiative_bc_pt<pw , nx>(&Gt_rhs2[gidx], Gt2[gidx], grad_0_Gt2,
    grad_1_Gt2, grad_2_Gt2, 2.0, 0.0, blk);

        radiative_bc_pt<pw , nx>(&B_rhs0[gidx], B0[gidx], grad_0_B0, grad_1_B0,
    grad_2_B0, 1.0, 0.0, blk); radiative_bc_pt<pw , nx>(&B_rhs1[gidx], B1[gidx],
    grad_0_B1, grad_1_B1, grad_2_B1, 1.0, 0.0, blk); radiative_bc_pt<pw ,
    nx>(&B_rhs2[gidx], B2[gidx], grad_0_B2, grad_1_B2, grad_2_B2, 1.0, 0.0,
    blk);

        radiative_bc_pt<pw , nx>(&At_rhs00[gidx], At0[gidx], grad_0_At0,
    grad_1_At0, grad_2_At0, 2.0, 0.0, blk); radiative_bc_pt<pw ,
    nx>(&At_rhs01[gidx], At1[gidx], grad_0_At1, grad_1_At1, grad_2_At1, 2.0,
    0.0, blk); radiative_bc_pt<pw , nx>(&At_rhs02[gidx], At2[gidx], grad_0_At2,
    grad_1_At2, grad_2_At2, 2.0, 0.0, blk); radiative_bc_pt<pw ,
    nx>(&At_rhs11[gidx], At3[gidx], grad_0_At3, grad_1_At3, grad_2_At3, 2.0,
    0.0, blk); radiative_bc_pt<pw , nx>(&At_rhs12[gidx], At4[gidx], grad_0_At4,
    grad_1_At4, grad_2_At4, 2.0, 0.0, blk); radiative_bc_pt<pw ,
    nx>(&At_rhs22[gidx], At5[gidx], grad_0_At5, grad_1_At5, grad_2_At5, 2.0,
    0.0, blk);

        radiative_bc_pt<pw , nx>(&gt_rhs00[gidx], gt0[gidx], grad_0_gt0,
    grad_1_gt0, grad_2_gt0, 1.0, 1.0, blk); radiative_bc_pt<pw ,
    nx>(&gt_rhs01[gidx], gt1[gidx], grad_0_gt1, grad_1_gt1, grad_2_gt1, 1.0,
    0.0, blk); radiative_bc_pt<pw , nx>(&gt_rhs02[gidx], gt2[gidx], grad_0_gt2,
    grad_1_gt2, grad_2_gt2, 1.0, 0.0, blk); radiative_bc_pt<pw ,
    nx>(&gt_rhs11[gidx], gt3[gidx], grad_0_gt3, grad_1_gt3,
    grad_2_gt3, 1.0, 1.0, blk); radiative_bc_pt<pw , nx>(&gt_rhs12[gidx],
    gt4[gidx], grad_0_gt4, grad_1_gt4, grad_2_gt4, 1.0, 0.0, blk);
        radiative_bc_pt<pw , nx>(&gt_rhs22[gidx], gt5[gidx], grad_0_gt5,
    grad_1_gt5, grad_2_gt5, 1.0, 1.0, blk);
    }

    GPUDevice::sync_threads();
    const DEVICE_REAL sigma = ko_sigma;
    a_rhs[pp]    += sigma * (kograd_0_alpha + kograd_1_alpha + kograd_2_alpha);
    b_rhs0[pp]   += sigma * (kograd_0_beta0 + kograd_1_beta0 + kograd_2_beta0);
    b_rhs1[pp]   += sigma * (kograd_0_beta1 + kograd_1_beta1 + kograd_2_beta1);
    b_rhs2[pp]   += sigma * (kograd_0_beta2 + kograd_1_beta2 + kograd_2_beta2);

    gt_rhs00[pp] += sigma * (kograd_0_gt0 + kograd_1_gt0 + kograd_2_gt0);
    gt_rhs01[pp] += sigma * (kograd_0_gt1 + kograd_1_gt1 + kograd_2_gt1);
    gt_rhs02[pp] += sigma * (kograd_0_gt2 + kograd_1_gt2 + kograd_2_gt2);
    gt_rhs11[pp] += sigma * (kograd_0_gt3 + kograd_1_gt3 + kograd_2_gt3);
    gt_rhs12[pp] += sigma * (kograd_0_gt4 + kograd_1_gt4 + kograd_2_gt4);
    gt_rhs22[pp] += sigma * (kograd_0_gt5 + kograd_1_gt5 + kograd_2_gt5);

    chi_rhs[pp]  += sigma * (kograd_0_chi + kograd_1_chi + kograd_2_chi);

    At_rhs00[pp] += sigma * (kograd_0_At0 + kograd_1_At0 + kograd_2_At0);
    At_rhs01[pp] += sigma * (kograd_0_At1 + kograd_1_At1 + kograd_2_At1);
    At_rhs02[pp] += sigma * (kograd_0_At2 + kograd_1_At2 + kograd_2_At2);
    At_rhs11[pp] += sigma * (kograd_0_At3 + kograd_1_At3 + kograd_2_At3);
    At_rhs12[pp] += sigma * (kograd_0_At4 + kograd_1_At4 + kograd_2_At4);
    At_rhs22[pp] += sigma * (kograd_0_At5 + kograd_1_At5 + kograd_2_At5);

    K_rhs[pp]    += sigma * (kograd_0_K + kograd_1_K + kograd_2_K);

    Gt_rhs0[pp]  += sigma * (kograd_0_Gt0 + kograd_1_Gt0 + kograd_2_Gt0);
    Gt_rhs1[pp]  += sigma * (kograd_0_Gt1 + kograd_1_Gt1 + kograd_2_Gt1);
    Gt_rhs2[pp]  += sigma * (kograd_0_Gt2 + kograd_1_Gt2 + kograd_2_Gt2);

    B_rhs0[pp]   += sigma * (kograd_0_B0 + kograd_1_B0 + kograd_2_B0);
    B_rhs1[pp]   += sigma * (kograd_0_B1 + kograd_1_B1 + kograd_2_B1);
    B_rhs2[pp]   += sigma * (kograd_0_B2 + kograd_1_B2 + kograd_2_B2);*/

    return;
}

// ---- Point-wise fused FD (items 1–2): one su buffer, no Du/DDu passes ----
// PreferL1 + lb(343,3): 0.137s @2048×20 (vs 0.171 baseline). lb(343,4) spilled more → 0.182s.
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_pt(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    extern __shared__ DEVICE_REAL shared_mem[];
    DEVICE_REAL* su = shared_mem;
    // Du/DDu unused — kept as nullptr aliases so any leftover refs fail loudly
    // at compile time rather than silently. gauge6_point.cpp does not use them.
    (void)pencil_sz;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    // PT fused FD is interior-only; domain-boundary blocks need stock
    // one-sided 644/KO stencils (see __blk1_deriv644_* bflag overrides).
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;
    (void)actual_nx;
    (void)BLK_SZ;
    (void)pencils;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i  = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j  = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k  = GPUDevice::thread_id_z() + pw;
    const int gidx      = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

#include "../scripts/gauge6_point.cpp"
}

// Global-memory point FD (no shared tile). NCU: PT_ASYNC smem-bound at 1
// blk/SM (17% occ). Zero smem → register limit (~3 blk/SM).
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_pt_gmem(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    (void)pencil_sz;
    (void)nblocks;
    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset = blk->m_offset;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i  = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j  = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k  = GPUDevice::thread_id_z() + pw;
    const int gidx      = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

#include "../scripts/gauge6_point_gmem.cpp"
}

// PT fused FD on centered float shared tile only (~8.8KB).
// PreferShared → register-limited 3 blk/SM (~45% occ). lb minBlocks=4 spills worse.
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 1) void eval_rhs3_pt_f32su(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    extern __shared__ DEVICE_REAL shared_mem[];
    float* su = reinterpret_cast<float*>(shared_mem);
    (void)pencil_sz;
    (void)nblocks;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset = blk->m_offset;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i  = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j  = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k  = GPUDevice::thread_id_z() + pw;
    const int gidx      = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

#include "../scripts/gauge6_point_f32su.cpp"
}

// cp.async load (issue+wait, no double-buffer overlap yet).
// K_share==0: v1 ensemble via grid.y (one CTA per (block,member)).
// K_share>=1: Tier 1.3 v2 — one CTA per mesh block; loop members, reuse
// block geometry / dx / eta (still K algebra + smem tile passes).
template <int pw, int nx, int pencil_sz>
// minBlocks=1: smem already caps occ at ~1 blk/SM (17.6KB); minBlocks=3
// forced a tighter reg budget and worsened local spills (B0 NCU).
GLOBAL_FUNC __launch_bounds__(343, 1) void eval_rhs3_pt_async(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma, DEVICE_UINT K_share) {
    extern __shared__ DEVICE_REAL shared_mem[];
    DEVICE_REAL* su = shared_mem;
    // Du/DDu unused — kept as nullptr aliases so any leftover refs fail loudly
    // at compile time rather than silently. gauge6_point_async.cpp does not use them.
    (void)pencil_sz;
    (void)nblocks;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    // PT fused FD is interior-only; domain-boundary blocks need stock
    // one-sided 644/KO stencils (see __blk1_deriv644_* bflag overrides).
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;
    (void)actual_nx;
    (void)BLK_SZ;
    (void)pencils;

    const DEVICE_INT i  = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j  = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k  = GPUDevice::thread_id_z() + pw;
    const int gidx      = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

    const DEVICE_UINT Kloop =
        (K_share > 0) ? K_share : static_cast<DEVICE_UINT>(1);
    const DEVICE_UINT kbase =
        (K_share > 0) ? static_cast<DEVICE_UINT>(0) : GPUDevice::block_id_y();

    for (DEVICE_UINT ki = 0; ki < Kloop; ++ki) {
        const DEVICE_UINT kmem = (K_share > 0) ? ki : kbase;
        DEVICE_REAL* const Fu =
            dptr_Fu + kmem * bssn::BSSN_NUM_VARS * sz_per_dof;
        const DEVICE_REAL* const u =
            dptr_u + kmem * bssn::BSSN_NUM_VARS * sz_per_dof;

        DEVICE_REAL* __restrict__ const a_rhs =
            &Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const chi_rhs =
            &Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs00 =
            &Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs01 =
            &Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs02 =
            &Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs11 =
            &Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs12 =
            &Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs22 =
            &Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs0 =
            &Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs1 =
            &Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs2 =
            &Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs00 =
            &Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs01 =
            &Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs02 =
            &Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs11 =
            &Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs12 =
            &Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs22 =
            &Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs0 =
            &Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs1 =
            &Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs2 =
            &Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs0 =
            &Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs1 =
            &Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs2 =
            &Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const K_rhs =
            &Fu[bssn::VAR::U_K * sz_per_dof + offset];

        const DEVICE_REAL* __restrict__ const alpha =
            (u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const chi =
            (u + bssn::VAR::U_CHI * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const K =
            (u + bssn::VAR::U_K * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt0 =
            (u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt1 =
            (u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt2 =
            (u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt3 =
            (u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt4 =
            (u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt5 =
            (u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta0 =
            (u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta1 =
            (u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta2 =
            (u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At0 =
            (u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At1 =
            (u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At2 =
            (u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At3 =
            (u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At4 =
            (u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At5 =
            (u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt0 =
            (u + bssn::VAR::U_GT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt1 =
            (u + bssn::VAR::U_GT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt2 =
            (u + bssn::VAR::U_GT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B0 =
            (u + bssn::VAR::U_B0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B1 =
            (u + bssn::VAR::U_B1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B2 =
            (u + bssn::VAR::U_B2 * sz_per_dof + offset);

#include "../scripts/gauge6_point_async.cpp"
        __syncthreads();  // separate member passes that reuse su
    }
}

// B1 deferred-DD seconds (reload before K/Gt/Ricci). Same launch as pt_async.
// K_share==0: v1 ensemble via grid.y (one CTA per (block,member)).
// K_share>=1: Tier 1.3 v2 — one CTA per mesh block; loop members, reuse
// block geometry / dx / eta (still K algebra + smem tile passes).
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 1) void eval_rhs3_pt_async_defer(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma, DEVICE_UINT K_share) {
    extern __shared__ DEVICE_REAL shared_mem[];
    DEVICE_REAL* su = shared_mem;
    // Du/DDu unused — kept as nullptr aliases so any leftover refs fail loudly
    // at compile time rather than silently. gauge6_point_async.cpp does not use them.
    (void)pencil_sz;
    (void)nblocks;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    // PT fused FD is interior-only; domain-boundary blocks need stock
    // one-sided 644/KO stencils (see __blk1_deriv644_* bflag overrides).
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;
    (void)actual_nx;
    (void)BLK_SZ;
    (void)pencils;

    const DEVICE_INT i  = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j  = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k  = GPUDevice::thread_id_z() + pw;
    const int gidx      = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

    const DEVICE_UINT Kloop =
        (K_share > 0) ? K_share : static_cast<DEVICE_UINT>(1);
    const DEVICE_UINT kbase =
        (K_share > 0) ? static_cast<DEVICE_UINT>(0) : GPUDevice::block_id_y();

    for (DEVICE_UINT ki = 0; ki < Kloop; ++ki) {
        const DEVICE_UINT kmem = (K_share > 0) ? ki : kbase;
        DEVICE_REAL* const Fu =
            dptr_Fu + kmem * bssn::BSSN_NUM_VARS * sz_per_dof;
        const DEVICE_REAL* const u =
            dptr_u + kmem * bssn::BSSN_NUM_VARS * sz_per_dof;

        DEVICE_REAL* __restrict__ const a_rhs =
            &Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const chi_rhs =
            &Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs00 =
            &Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs01 =
            &Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs02 =
            &Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs11 =
            &Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs12 =
            &Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs22 =
            &Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs0 =
            &Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs1 =
            &Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs2 =
            &Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs00 =
            &Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs01 =
            &Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs02 =
            &Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs11 =
            &Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs12 =
            &Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs22 =
            &Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs0 =
            &Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs1 =
            &Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs2 =
            &Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs0 =
            &Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs1 =
            &Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs2 =
            &Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const K_rhs =
            &Fu[bssn::VAR::U_K * sz_per_dof + offset];

        const DEVICE_REAL* __restrict__ const alpha =
            (u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const chi =
            (u + bssn::VAR::U_CHI * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const K =
            (u + bssn::VAR::U_K * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt0 =
            (u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt1 =
            (u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt2 =
            (u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt3 =
            (u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt4 =
            (u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt5 =
            (u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta0 =
            (u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta1 =
            (u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta2 =
            (u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At0 =
            (u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At1 =
            (u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At2 =
            (u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At3 =
            (u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At4 =
            (u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At5 =
            (u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt0 =
            (u + bssn::VAR::U_GT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt1 =
            (u + bssn::VAR::U_GT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt2 =
            (u + bssn::VAR::U_GT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B0 =
            (u + bssn::VAR::U_B0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B1 =
            (u + bssn::VAR::U_B1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B2 =
            (u + bssn::VAR::U_B2 * sz_per_dof + offset);

#include "../scripts/gauge6_point_async_defer.cpp"
        __syncthreads();  // separate member passes that reuse su
    }
}

template <int pw, int nx, int pencil_sz>
// B1 CSE: single christoffel (igt/C*); longer live ranges. A/B vs async.
GLOBAL_FUNC __launch_bounds__(343, 1) void eval_rhs3_pt_async_cse(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma, DEVICE_UINT K_share) {
    extern __shared__ DEVICE_REAL shared_mem[];
    DEVICE_REAL* su = shared_mem;
    // Du/DDu unused — kept as nullptr aliases so any leftover refs fail loudly
    // at compile time rather than silently. gauge6_point_async_cse.cpp does not use them.
    (void)pencil_sz;
    (void)nblocks;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    // PT fused FD is interior-only; domain-boundary blocks need stock
    // one-sided 644/KO stencils (see __blk1_deriv644_* bflag overrides).
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;
    (void)actual_nx;
    (void)BLK_SZ;
    (void)pencils;

    const DEVICE_INT i  = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j  = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k  = GPUDevice::thread_id_z() + pw;
    const int gidx      = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

    const DEVICE_UINT Kloop =
        (K_share > 0) ? K_share : static_cast<DEVICE_UINT>(1);
    const DEVICE_UINT kbase =
        (K_share > 0) ? static_cast<DEVICE_UINT>(0) : GPUDevice::block_id_y();

    for (DEVICE_UINT ki = 0; ki < Kloop; ++ki) {
        const DEVICE_UINT kmem = (K_share > 0) ? ki : kbase;
        DEVICE_REAL* const Fu =
            dptr_Fu + kmem * bssn::BSSN_NUM_VARS * sz_per_dof;
        const DEVICE_REAL* const u =
            dptr_u + kmem * bssn::BSSN_NUM_VARS * sz_per_dof;

        DEVICE_REAL* __restrict__ const a_rhs =
            &Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const chi_rhs =
            &Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs00 =
            &Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs01 =
            &Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs02 =
            &Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs11 =
            &Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs12 =
            &Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs22 =
            &Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs0 =
            &Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs1 =
            &Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs2 =
            &Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs00 =
            &Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs01 =
            &Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs02 =
            &Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs11 =
            &Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs12 =
            &Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs22 =
            &Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs0 =
            &Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs1 =
            &Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs2 =
            &Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs0 =
            &Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs1 =
            &Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs2 =
            &Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const K_rhs =
            &Fu[bssn::VAR::U_K * sz_per_dof + offset];

        const DEVICE_REAL* __restrict__ const alpha =
            (u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const chi =
            (u + bssn::VAR::U_CHI * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const K =
            (u + bssn::VAR::U_K * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt0 =
            (u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt1 =
            (u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt2 =
            (u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt3 =
            (u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt4 =
            (u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt5 =
            (u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta0 =
            (u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta1 =
            (u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta2 =
            (u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At0 =
            (u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At1 =
            (u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At2 =
            (u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At3 =
            (u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At4 =
            (u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At5 =
            (u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt0 =
            (u + bssn::VAR::U_GT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt1 =
            (u + bssn::VAR::U_GT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt2 =
            (u + bssn::VAR::U_GT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B0 =
            (u + bssn::VAR::U_B0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B1 =
            (u + bssn::VAR::U_B1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B2 =
            (u + bssn::VAR::U_B2 * sz_per_dof + offset);

#include "../scripts/gauge6_point_async_cse.cpp"
        __syncthreads();  // separate member passes that reuse su
    }
}

template <int pw, int nx, int pencil_sz>
// B1 while-hot: early RIJ + scoped gt/chi seconds (no restencil).
GLOBAL_FUNC __launch_bounds__(343, 1) void eval_rhs3_pt_async_whilehot(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma, DEVICE_UINT K_share) {
    extern __shared__ DEVICE_REAL shared_mem[];
    DEVICE_REAL* su = shared_mem;
    // Du/DDu unused — kept as nullptr aliases so any leftover refs fail loudly
    // at compile time rather than silently. gauge6_point_async_whilehot.cpp does not use them.
    (void)pencil_sz;
    (void)nblocks;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    // PT fused FD is interior-only; domain-boundary blocks need stock
    // one-sided 644/KO stencils (see __blk1_deriv644_* bflag overrides).
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;
    (void)actual_nx;
    (void)BLK_SZ;
    (void)pencils;

    const DEVICE_INT i  = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j  = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k  = GPUDevice::thread_id_z() + pw;
    const int gidx      = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

    const DEVICE_UINT Kloop =
        (K_share > 0) ? K_share : static_cast<DEVICE_UINT>(1);
    const DEVICE_UINT kbase =
        (K_share > 0) ? static_cast<DEVICE_UINT>(0) : GPUDevice::block_id_y();

    for (DEVICE_UINT ki = 0; ki < Kloop; ++ki) {
        const DEVICE_UINT kmem = (K_share > 0) ? ki : kbase;
        DEVICE_REAL* const Fu =
            dptr_Fu + kmem * bssn::BSSN_NUM_VARS * sz_per_dof;
        const DEVICE_REAL* const u =
            dptr_u + kmem * bssn::BSSN_NUM_VARS * sz_per_dof;

        DEVICE_REAL* __restrict__ const a_rhs =
            &Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const chi_rhs =
            &Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs00 =
            &Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs01 =
            &Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs02 =
            &Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs11 =
            &Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs12 =
            &Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const gt_rhs22 =
            &Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs0 =
            &Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs1 =
            &Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const b_rhs2 =
            &Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs00 =
            &Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs01 =
            &Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs02 =
            &Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs11 =
            &Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs12 =
            &Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const At_rhs22 =
            &Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs0 =
            &Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs1 =
            &Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const Gt_rhs2 =
            &Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs0 =
            &Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs1 =
            &Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const B_rhs2 =
            &Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
        DEVICE_REAL* __restrict__ const K_rhs =
            &Fu[bssn::VAR::U_K * sz_per_dof + offset];

        const DEVICE_REAL* __restrict__ const alpha =
            (u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const chi =
            (u + bssn::VAR::U_CHI * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const K =
            (u + bssn::VAR::U_K * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt0 =
            (u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt1 =
            (u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt2 =
            (u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt3 =
            (u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt4 =
            (u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const gt5 =
            (u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta0 =
            (u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta1 =
            (u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const beta2 =
            (u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At0 =
            (u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At1 =
            (u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At2 =
            (u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At3 =
            (u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At4 =
            (u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const At5 =
            (u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt0 =
            (u + bssn::VAR::U_GT0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt1 =
            (u + bssn::VAR::U_GT1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const Gt2 =
            (u + bssn::VAR::U_GT2 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B0 =
            (u + bssn::VAR::U_B0 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B1 =
            (u + bssn::VAR::U_B1 * sz_per_dof + offset);
        const DEVICE_REAL* __restrict__ const B2 =
            (u + bssn::VAR::U_B2 * sz_per_dof + offset);

#include "../scripts/gauge6_point_async_whilehot.cpp"
        __syncthreads();  // separate member passes that reuse su
    }
}

// Double-buffered cp.async: prefetch v+1 into idle su while algebra for v runs.
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_pt_db(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    extern __shared__ DEVICE_REAL shared_mem[];
    DEVICE_REAL* su = shared_mem;  // su0; su1 = su + nx^3 (see gauge6_point_db)
    (void)pencil_sz;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    // PT fused FD is interior-only; domain-boundary blocks need stock
    // one-sided 644/KO stencils (see __blk1_deriv644_* bflag overrides).
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;
    (void)actual_nx;
    (void)BLK_SZ;
    (void)pencils;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i      = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j      = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k      = GPUDevice::thread_id_z() + pw;
    const int gidx          = (k * nx + j) * nx + i;
    const DEVICE_INT pp     = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

#include "../scripts/gauge6_point_db.cpp"
}

// Point-wise + KO dissipation in FP32 (item 6b).
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_pt_kof32(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    extern __shared__ DEVICE_REAL shared_mem[];
    DEVICE_REAL* su = shared_mem;
    // Du/DDu unused — kept as nullptr aliases so any leftover refs fail loudly
    // at compile time rather than silently. gauge6_point.cpp does not use them.
    (void)pencil_sz;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    // PT fused FD is interior-only; domain-boundary blocks need stock
    // one-sided 644/KO stencils (see __blk1_deriv644_* bflag overrides).
    if (blk->m_bflag != 0) return;

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;
    (void)actual_nx;
    (void)BLK_SZ;
    (void)pencils;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i  = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j  = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k  = GPUDevice::thread_id_z() + pw;
    const int gidx      = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x  = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y  = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z  = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;
    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

#define PT_FUSED_X3 device::__pt_fused_x3_kof32
#define PT_FUSED_Y3 device::__pt_fused_y3_kof32
#define PT_FUSED_Z3 device::__pt_fused_z3_kof32
#include "../scripts/gauge6_point.cpp"
#undef PT_FUSED_X3
#undef PT_FUSED_Y3
#undef PT_FUSED_Z3
}

// ---- FP32 Du/DDu shared scratch (FP64 state + algebra regs) ----
// Note: short evolve shows C_HAM L2 growth vs FP64 — keep behind DENDRO_RHS_F32.
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_f32(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    // Same extern __shared__ type as other kernels in this TU (CUDA rule).
    extern __shared__ DEVICE_REAL shared_mem[];
    const int n3 = pencil_sz * pencil_sz * pencil_sz;
    float* su  = reinterpret_cast<float*>(shared_mem);
    float* Du  = su + n3;
    float* DDu = Du + n3;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    if (blk->m_bflag != 0) return;  // interior FP32 stencils only

    const DEVICE_UINT offset                 = blk->m_offset;
    const int pencils                        = pencil_sz;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k = GPUDevice::thread_id_z() + pw;
    const int gidx = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH = 40.0;
    const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;
#include "../scripts/gauge6_f32.cpp"
}

// Forward decl: defined with PT host launchers (hybrid face blocks).
template <int pw, int nx, int pencil_sz>
HOST_FUNC inline void bssnrhs3_stock_boundary(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma, cudaStream_t bdy_stream = 0);

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_f32(DEVICE_REAL* const dptr_Fu,
                            const DEVICE_REAL* const dptr_u, BlockGPU3D* dptr_blk,
                            DEVICE_UINT nblocks, size_t szpdof_uz,
                            DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 4 * 3;
    cudaFuncSetAttribute(device::eval_rhs3_f32<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    // Measured: PreferShared 0.142s vs PreferL1 0.181s @2048×20.
    // Override with DENDRO_F32_PREFER_L1=1.
    if (std::getenv("DENDRO_F32_PREFER_L1"))
        cudaFuncSetCacheConfig(device::eval_rhs3_f32<pw, nx, pencil_sz>,
                               cudaFuncCachePreferL1);
    else
        cudaFuncSetCacheConfig(device::eval_rhs3_f32<pw, nx, pencil_sz>,
                               cudaFuncCachePreferShared);
    eval_rhs3_f32<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    // Interior-only FP32 stencils: domain faces use stock one-sided FD.
    bssnrhs3_stock_boundary<pw, nx, pencil_sz>(dptr_Fu, dptr_u, dptr_blk,
                                               nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();
}


// ---- FP32 Du/DDu shared scratch (FP64 state + algebra regs) ----
// Centered float cast of (u-u_c) into su; expect better puncture accuracy than raw f32.
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_f32c(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    // Same extern __shared__ type as other kernels in this TU (CUDA rule).
    extern __shared__ DEVICE_REAL shared_mem[];
    const int n3 = pencil_sz * pencil_sz * pencil_sz;
    float* su  = reinterpret_cast<float*>(shared_mem);
    float* Du  = su + n3;
    float* DDu = Du + n3;

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    if (blk->m_bflag != 0) return;  // interior FP32 stencils only

    const DEVICE_UINT offset                 = blk->m_offset;
    const int pencils                        = pencil_sz;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k = GPUDevice::thread_id_z() + pw;
    const int gidx = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH = 40.0;
    const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;
#include "../scripts/gauge6_f32c.cpp"
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_f32c(DEVICE_REAL* const dptr_Fu,
                            const DEVICE_REAL* const dptr_u, BlockGPU3D* dptr_blk,
                            DEVICE_UINT nblocks, size_t szpdof_uz,
                            DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 4 * 3;
    cudaFuncSetAttribute(device::eval_rhs3_f32c<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    // Measured: PreferShared 0.142s vs PreferL1 0.181s @2048×20.
    // Override with DENDRO_F32_PREFER_L1=1.
    if (std::getenv("DENDRO_F32_PREFER_L1"))
        cudaFuncSetCacheConfig(device::eval_rhs3_f32c<pw, nx, pencil_sz>,
                               cudaFuncCachePreferL1);
    else
        cudaFuncSetCacheConfig(device::eval_rhs3_f32c<pw, nx, pencil_sz>,
                               cudaFuncCachePreferShared);
    eval_rhs3_f32c<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    bssnrhs3_stock_boundary<pw, nx, pencil_sz>(dptr_Fu, dptr_u, dptr_blk,
                                               nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();
}


// ---- Hybrid: float su + FP64 Du/DDu ----
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_f2d(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    extern __shared__ DEVICE_REAL shared_mem[];
    const int n3 = pencil_sz * pencil_sz * pencil_sz;
    // layout: float su[n3] | pad to 8B | double Du[n3] | double DDu[n3]
    float* su = reinterpret_cast<float*>(shared_mem);
    const int su_bytes = (n3 * (int)sizeof(float) + 7) & ~7;
    DEVICE_REAL* Du = reinterpret_cast<DEVICE_REAL*>(
        reinterpret_cast<char*>(shared_mem) + su_bytes);
    DEVICE_REAL* DDu = Du + n3;

    const size_t sz_per_dof = szpdof_uz;
    const DEVICE_UINT lambda[4] = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2] = {1.0, 0.0};
    const DEVICE_INT Z_ID = GPUDevice::block_id_x();
    BlockGPU3D c_blk = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    const DEVICE_UINT offset = blk->m_offset;
    const int pencils = pencil_sz;

    DEVICE_REAL* __restrict__ const a_rhs = &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs = &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 = &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 = &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 = &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 = &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 = &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 = &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 = &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 = &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 = &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 = &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 = &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 = &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 = &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 = &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 = &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 = &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 = &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 = &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 = &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 = &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 = &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs = &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha = (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi = (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K = (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 = (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 = (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 = (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 = (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 = (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 = (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 = (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 = (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 = (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 = (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 = (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 = (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 = (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 = (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 = (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 = (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 = (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 = (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 = (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 = (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 = (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k = GPUDevice::thread_id_z() + pw;
    const int gidx = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;
    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH = 40.0;
    const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg = -w * w * w * w;
    const DEVICE_REAL eta = (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;
#include "../scripts/gauge6_f2d.cpp"
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_f2d(DEVICE_REAL* const dptr_Fu,
                            const DEVICE_REAL* const dptr_u, BlockGPU3D* dptr_blk,
                            DEVICE_UINT nblocks, size_t szpdof_uz,
                            DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    const int n3 = 13 * 13 * 13;
    const int su_bytes = (n3 * (int)sizeof(float) + 7) & ~7;
    const int smem = su_bytes + 2 * n3 * (int)sizeof(DEVICE_REAL);
    cudaFuncSetAttribute(device::eval_rhs3_f2d<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetCacheConfig(device::eval_rhs3_f2d<pw, nx, pencil_sz>,
                           cudaFuncCachePreferShared);
    eval_rhs3_f2d<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();
}

// ---- Interior-workspace 2-kernel path (7^3 grads / block) ----
// Optional: requires gauge6_derivs_ws.cpp + gauge6_algebra_ws.cpp (not in
// the GB200 transfer tarball; A100 already showed WS slower than PT).
#if __has_include("../scripts/gauge6_derivs_ws.cpp") && \
    __has_include("../scripts/gauge6_algebra_ws.cpp")
#define DENDRO_HAS_WS_KERNELS 1
template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_derivs_ws(
    const DEVICE_REAL* const dptr_u, const BlockGPU3D* dptr_blk,
    DEVICE_UINT nblocks, size_t szpdof_uz, DEVICE_REAL* grad_ws) {
    extern __shared__ DEVICE_REAL shared_mem[];
    DEVICE_REAL* su  = shared_mem;
    DEVICE_REAL* Du  = shared_mem + pencil_sz * pencil_sz * pencil_sz;
    DEVICE_REAL* DDu = shared_mem + 2 * pencil_sz * pencil_sz * pencil_sz;

    constexpr int INTERIOR = 7 * 7 * 7;
    const DEVICE_UINT nblocks_ws = nblocks;
    const DEVICE_INT Z_ID = GPUDevice::block_id_x();
    BlockGPU3D c_blk = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    const DEVICE_UINT offset = blk->m_offset;
    const size_t sz_per_dof = szpdof_uz;
    const int pencils = pencil_sz;

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k = GPUDevice::thread_id_z() + pw;
    const int gidx = (k * nx + j) * nx + i;
    const int tid = GPUDevice::thread_id_x() +
                    7 * (GPUDevice::thread_id_y() + 7 * GPUDevice::thread_id_z());
#include "../scripts/gauge6_derivs_ws.cpp"
}

template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs3_algebra_ws(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma, const DEVICE_REAL* grad_ws) {
    constexpr int INTERIOR = 7 * 7 * 7;
    const DEVICE_UINT nblocks_ws = nblocks;
    const DEVICE_UINT lambda[4] = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2] = {1.0, 0.0};
    const DEVICE_INT Z_ID = GPUDevice::block_id_x();
    BlockGPU3D c_blk = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    const DEVICE_UINT offset = blk->m_offset;
    const size_t sz_per_dof = szpdof_uz;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k = GPUDevice::thread_id_z() + pw;
    const int gidx = (k * nx + j) * nx + i;
    const DEVICE_INT pp = gidx;
    const int tid = GPUDevice::thread_id_x() +
                    7 * (GPUDevice::thread_id_y() + 7 * GPUDevice::thread_id_z());

    const DEVICE_REAL hx = blk->m_dx[0];
    const DEVICE_REAL x = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z = blk->m_ptMin[2] + k * blk->m_dx[2];
    const DEVICE_REAL RIT_ETA_OUTER = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH = 40.0;
    const DEVICE_REAL r_coord = sqrt(x * x + y * y + z * z);
    const DEVICE_REAL w = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;
#include "../scripts/gauge6_algebra_ws.cpp"
}

template <int pw, int pencil_sz, int nx>
HOST_FUNC void bssnrhs3_ws(DEVICE_REAL* const dptr_Fu,
                           const DEVICE_REAL* const dptr_u, BlockGPU3D* dptr_blk,
                           DEVICE_UINT nblocks, size_t szpdof_uz,
                           DEVICE_REAL ko_sigma, DEVICE_REAL* grad_ws) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 8 * 3;
    cudaFuncSetAttribute(device::eval_rhs3_derivs_ws<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetCacheConfig(device::eval_rhs3_derivs_ws<pw, nx, pencil_sz>,
                           cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(device::eval_rhs3_algebra_ws<pw, nx, pencil_sz>,
                           cudaFuncCachePreferL1);
    eval_rhs3_derivs_ws<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_u, dptr_blk, nblocks, szpdof_uz, grad_ws);
    eval_rhs3_algebra_ws<pw, nx, pencil_sz><<<gb, tb>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma, grad_ws);
    cudaDeviceSynchronize();
}
#else
#define DENDRO_HAS_WS_KERNELS 0
#endif  // DENDRO_HAS_WS_KERNELS (optional generated WS scripts)

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3(DEVICE_REAL* const dptr_Fu,
                        const DEVICE_REAL* const dptr_u, BlockGPU3D* dptr_blk,
                        DEVICE_UINT nblocks, size_t szpdof_uz,
                        DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);

    cudaFuncSetCacheConfig(device::eval_rhs3<pw, nx, pencil_sz>,
                           cudaFuncCachePreferShared);
    eval_rhs3<pw, nx, pencil_sz><<<gb, tb, 13 * 13 * 13 * 8 * 3>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();

    return;
}

// Device compact launcher (kernel lives in compact_bdy.cu — one TU).
HOST_FUNC void launch_compact_bdy_blocks(const BlockGPU3D* in, BlockGPU3D* out,
                                         DEVICE_UINT* count, DEVICE_UINT n);

// Hybrid: PT (or async/db) for interior blocks; stock eval_rhs3 for
// domain-boundary blocks that need one-sided FD (bflag!=0).
// Tier 2.3: DENDRO_COMPACT_BDY=1 → device-side compact (no host D2H).
template <int pw, int nx, int pencil_sz>
HOST_FUNC inline void bssnrhs3_stock_boundary(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    BlockGPU3D* dptr_blk, DEVICE_UINT nblocks, size_t szpdof_uz,
    DEVICE_REAL ko_sigma, cudaStream_t bdy_stream) {
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 8 * 3;
    const char* compact_env = std::getenv("DENDRO_COMPACT_BDY");
    const bool compact =
        compact_env && (compact_env[0] == '1') && compact_env[1] == '\0';

    BlockGPU3D* launch_blk = dptr_blk;
    DEVICE_UINT launch_n   = nblocks;
    static thread_local BlockGPU3D* d_bdy       = nullptr;
    static thread_local DEVICE_UINT* d_count    = nullptr;
    static thread_local DEVICE_UINT d_bdy_cap   = 0;
    static thread_local DEVICE_UINT cached_n    = 0;
    static thread_local DEVICE_UINT cached_nbdy = 0;
    static thread_local const BlockGPU3D* cached_src = nullptr;

    if (compact && nblocks > 0) {
        if (cached_src != dptr_blk || cached_n != nblocks) {
            if (nblocks > d_bdy_cap) {
                if (d_bdy) cudaFree(d_bdy);
                cudaMalloc(reinterpret_cast<void**>(&d_bdy),
                           nblocks * sizeof(BlockGPU3D));
                d_bdy_cap = nblocks;
            }
            if (!d_count) {
                cudaMalloc(reinterpret_cast<void**>(&d_count),
                           sizeof(DEVICE_UINT));
            }
            cudaMemset(d_count, 0, sizeof(DEVICE_UINT));
            launch_compact_bdy_blocks(dptr_blk, d_bdy, d_count, nblocks);
            cudaMemcpy(&cached_nbdy, d_count, sizeof(DEVICE_UINT),
                       cudaMemcpyDeviceToHost);
            cached_src = dptr_blk;
            cached_n   = nblocks;
            static thread_local bool logged = false;
            if (!logged) {
                printf("[COMPACT_BDY] device nblocks=%u nboundary=%u (%.1f%%)\n",
                       nblocks, cached_nbdy,
                       nblocks ? 100.0 * cached_nbdy / nblocks : 0.0);
                logged = true;
            }
        }
        if (cached_nbdy == 0) return;
        launch_blk = d_bdy;
        launch_n   = cached_nbdy;
    }

    dim3 gb = dim3(launch_n, 1, 1);
    if (compact) {
        cudaFuncSetAttribute(device::eval_rhs3<pw, nx, pencil_sz, 0>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem);
        cudaFuncSetCacheConfig(device::eval_rhs3<pw, nx, pencil_sz, 0>,
                               cudaFuncCachePreferShared);
        eval_rhs3<pw, nx, pencil_sz, 0><<<gb, tb, smem, bdy_stream>>>(
            dptr_Fu, dptr_u, launch_blk, launch_n, szpdof_uz, ko_sigma);
    } else {
        cudaFuncSetAttribute(device::eval_rhs3<pw, nx, pencil_sz, 2>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem);
        cudaFuncSetCacheConfig(device::eval_rhs3<pw, nx, pencil_sz, 2>,
                               cudaFuncCachePreferShared);
        eval_rhs3<pw, nx, pencil_sz, 2><<<gb, tb, smem, bdy_stream>>>(
            dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    }
}

// Point-wise fused derivs (items 1–2): su only, no Du/DDu block passes.
template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt(DEVICE_REAL* const dptr_Fu,
                           const DEVICE_REAL* const dptr_u,
                           BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                           size_t szpdof_uz, DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 8;  // su only
    cudaFuncSetAttribute(device::eval_rhs3_pt<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    // With ~18KB smem, PreferL1 helps spill traffic; override with
    // DENDRO_PT_PREFER_SHARED=1.
    if (std::getenv("DENDRO_PT_PREFER_SHARED"))
        cudaFuncSetCacheConfig(device::eval_rhs3_pt<pw, nx, pencil_sz>,
                               cudaFuncCachePreferShared);
    else
        cudaFuncSetCacheConfig(device::eval_rhs3_pt<pw, nx, pencil_sz>,
                               cudaFuncCachePreferL1);
    eval_rhs3_pt<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    bssnrhs3_stock_boundary<pw, nx, pencil_sz>(dptr_Fu, dptr_u, dptr_blk,
                                               nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt_async(DEVICE_REAL* const dptr_Fu,
                                 const DEVICE_REAL* const dptr_u,
                                 BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                                 size_t szpdof_uz, DEVICE_REAL ko_sigma,
                                 DEVICE_UINT K = 1, bool kshare = false) {
    // grid.x = shared block list; grid.y = ensemble members (leading dim on
    // unzipped vectors). K=1 preserves the prior single-member launch.
    // kshare: one CTA/block loops K members (Tier 1.3 v2 geometry reuse).
    if (K < 1) K = 1;
    dim3 gb = kshare ? dim3(nblocks, 1, 1) : dim3(nblocks, K, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 8;
    const DEVICE_UINT K_share = kshare ? K : 0;
    // DENDRO_BDY_STREAM=1: launch the (few-CTA, latency-bound) stock boundary
    // kernel FIRST on a side stream so it overlaps the interior kernel.
    // Kernels write disjoint blocks (interior: m_bflag==0, boundary: !=0), so
    // results are bit-identical to the serial order.
    static const bool bdy_overlap = []() {
        // DEFAULT ON (opt out with DENDRO_BDY_STREAM=0): measured -11% ETS,
        // GW output byte-identical (tier0/ab_bdystream.log).
        const char* e = std::getenv("DENDRO_BDY_STREAM");
        return !(e && e[0] == '0');
    }();
    static cudaStream_t s_bdy = nullptr;
    static cudaEvent_t ev_in  = nullptr;
    if (bdy_overlap && !s_bdy) {
        cudaStreamCreateWithFlags(&s_bdy, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&ev_in, cudaEventDisableTiming);
    }
    if (bdy_overlap) {  // order side stream after all prior default-stream work
        cudaEventRecord(ev_in, 0);
        cudaStreamWaitEvent(s_bdy, ev_in, 0);
    }
    cudaFuncSetAttribute(device::eval_rhs3_pt_async<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetCacheConfig(device::eval_rhs3_pt_async<pw, nx, pencil_sz>,
                           cudaFuncCachePreferL1);
    if (bdy_overlap) {
        for (DEVICE_UINT k = 0; k < K; ++k) {
            const size_t stride = (size_t)bssn::BSSN_NUM_VARS * szpdof_uz;
            bssnrhs3_stock_boundary<pw, nx, pencil_sz>(
                dptr_Fu + k * stride, dptr_u + k * stride, dptr_blk, nblocks,
                szpdof_uz, ko_sigma, s_bdy);
        }
    }
    eval_rhs3_pt_async<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma, K_share);
    if (!bdy_overlap) {
        for (DEVICE_UINT k = 0; k < K; ++k) {
            const size_t stride = (size_t)bssn::BSSN_NUM_VARS * szpdof_uz;
            bssnrhs3_stock_boundary<pw, nx, pencil_sz>(
                dptr_Fu + k * stride, dptr_u + k * stride, dptr_blk, nblocks,
                szpdof_uz, ko_sigma);
        }
    }
    cudaDeviceSynchronize();
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt_async_defer(DEVICE_REAL* const dptr_Fu,
                                 const DEVICE_REAL* const dptr_u,
                                 BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                                 size_t szpdof_uz, DEVICE_REAL ko_sigma,
                                 DEVICE_UINT K = 1, bool kshare = false) {
    // grid.x = shared block list; grid.y = ensemble members (leading dim on
    // unzipped vectors). K=1 preserves the prior single-member launch.
    // kshare: one CTA/block loops K members (Tier 1.3 v2 geometry reuse).
    if (K < 1) K = 1;
    dim3 gb = kshare ? dim3(nblocks, 1, 1) : dim3(nblocks, K, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 8;
    const DEVICE_UINT K_share = kshare ? K : 0;
    cudaFuncSetAttribute(device::eval_rhs3_pt_async_defer<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetCacheConfig(device::eval_rhs3_pt_async_defer<pw, nx, pencil_sz>,
                           cudaFuncCachePreferL1);
    eval_rhs3_pt_async_defer<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma, K_share);
    for (DEVICE_UINT k = 0; k < K; ++k) {
        const size_t stride = (size_t)bssn::BSSN_NUM_VARS * szpdof_uz;
        bssnrhs3_stock_boundary<pw, nx, pencil_sz>(
            dptr_Fu + k * stride, dptr_u + k * stride, dptr_blk, nblocks,
            szpdof_uz, ko_sigma);
    }
    cudaDeviceSynchronize();
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt_async_cse(DEVICE_REAL* const dptr_Fu,
                                 const DEVICE_REAL* const dptr_u,
                                 BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                                 size_t szpdof_uz, DEVICE_REAL ko_sigma,
                                 DEVICE_UINT K = 1, bool kshare = false) {
    // grid.x = shared block list; grid.y = ensemble members (leading dim on
    // unzipped vectors). K=1 preserves the prior single-member launch.
    // kshare: one CTA/block loops K members (Tier 1.3 v2 geometry reuse).
    if (K < 1) K = 1;
    dim3 gb = kshare ? dim3(nblocks, 1, 1) : dim3(nblocks, K, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 8;
    const DEVICE_UINT K_share = kshare ? K : 0;
    cudaFuncSetAttribute(device::eval_rhs3_pt_async_cse<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetCacheConfig(device::eval_rhs3_pt_async_cse<pw, nx, pencil_sz>,
                           cudaFuncCachePreferL1);
    eval_rhs3_pt_async_cse<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma, K_share);
    for (DEVICE_UINT k = 0; k < K; ++k) {
        const size_t stride = (size_t)bssn::BSSN_NUM_VARS * szpdof_uz;
        bssnrhs3_stock_boundary<pw, nx, pencil_sz>(
            dptr_Fu + k * stride, dptr_u + k * stride, dptr_blk, nblocks,
            szpdof_uz, ko_sigma);
    }
    cudaDeviceSynchronize();
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt_async_whilehot(DEVICE_REAL* const dptr_Fu,
                                 const DEVICE_REAL* const dptr_u,
                                 BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                                 size_t szpdof_uz, DEVICE_REAL ko_sigma,
                                 DEVICE_UINT K = 1, bool kshare = false) {
    // grid.x = shared block list; grid.y = ensemble members (leading dim on
    // unzipped vectors). K=1 preserves the prior single-member launch.
    // kshare: one CTA/block loops K members (Tier 1.3 v2 geometry reuse).
    if (K < 1) K = 1;
    dim3 gb = kshare ? dim3(nblocks, 1, 1) : dim3(nblocks, K, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 8;
    const DEVICE_UINT K_share = kshare ? K : 0;
    cudaFuncSetAttribute(device::eval_rhs3_pt_async_whilehot<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetCacheConfig(device::eval_rhs3_pt_async_whilehot<pw, nx, pencil_sz>,
                           cudaFuncCachePreferL1);
    eval_rhs3_pt_async_whilehot<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma, K_share);
    for (DEVICE_UINT k = 0; k < K; ++k) {
        const size_t stride = (size_t)bssn::BSSN_NUM_VARS * szpdof_uz;
        bssnrhs3_stock_boundary<pw, nx, pencil_sz>(
            dptr_Fu + k * stride, dptr_u + k * stride, dptr_blk, nblocks,
            szpdof_uz, ko_sigma);
    }
    cudaDeviceSynchronize();
}

// No-shared global stencil path (occupancy-focused).
template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt_gmem(DEVICE_REAL* const dptr_Fu,
                                const DEVICE_REAL* const dptr_u,
                                BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                                size_t szpdof_uz, DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    cudaFuncSetCacheConfig(device::eval_rhs3_pt_gmem<pw, nx, pencil_sz>,
                           cudaFuncCachePreferL1);
    eval_rhs3_pt_gmem<pw, nx, pencil_sz><<<gb, tb>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    bssnrhs3_stock_boundary<pw, nx, pencil_sz>(dptr_Fu, dptr_u, dptr_blk,
                                               nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();
}

// PT fused + float shared tile only.
// PreferShared: PreferL1 carves smem so small that even 9KB → 1 blk/SM on GB200.
template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt_f32su(DEVICE_REAL* const dptr_Fu,
                                 const DEVICE_REAL* const dptr_u,
                                 BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                                 size_t szpdof_uz, DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 4;  // float su only
    cudaFuncSetAttribute(device::eval_rhs3_pt_f32su<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (std::getenv("DENDRO_F32SU_PREFER_L1"))
        cudaFuncSetCacheConfig(device::eval_rhs3_pt_f32su<pw, nx, pencil_sz>,
                               cudaFuncCachePreferL1);
    else
        cudaFuncSetCacheConfig(device::eval_rhs3_pt_f32su<pw, nx, pencil_sz>,
                               cudaFuncCachePreferShared);
    eval_rhs3_pt_f32su<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    bssnrhs3_stock_boundary<pw, nx, pencil_sz>(dptr_Fu, dptr_u, dptr_blk,
                                               nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt_db(DEVICE_REAL* const dptr_Fu,
                              const DEVICE_REAL* const dptr_u,
                              BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                              size_t szpdof_uz, DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    // Two su tiles (su0/su1) for overlapped prefetch.
    const int smem = 13 * 13 * 13 * 8 * 2;
    cudaFuncSetAttribute(device::eval_rhs3_pt_db<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    // Default PreferL1 (~35KB 2×su); override with DENDRO_PT_PREFER_SHARED=1.
    if (std::getenv("DENDRO_PT_PREFER_SHARED"))
        cudaFuncSetCacheConfig(device::eval_rhs3_pt_db<pw, nx, pencil_sz>,
                               cudaFuncCachePreferShared);
    else
        cudaFuncSetCacheConfig(device::eval_rhs3_pt_db<pw, nx, pencil_sz>,
                               cudaFuncCachePreferL1);
    eval_rhs3_pt_db<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    bssnrhs3_stock_boundary<pw, nx, pencil_sz>(dptr_Fu, dptr_u, dptr_blk,
                                               nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs3_pt_kof32(DEVICE_REAL* const dptr_Fu,
                                 const DEVICE_REAL* const dptr_u,
                                 BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
                                 size_t szpdof_uz, DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);
    const int smem = 13 * 13 * 13 * 8;
    cudaFuncSetAttribute(device::eval_rhs3_pt_kof32<pw, nx, pencil_sz>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetCacheConfig(device::eval_rhs3_pt_kof32<pw, nx, pencil_sz>,
                           cudaFuncCachePreferL1);
    eval_rhs3_pt_kof32<pw, nx, pencil_sz><<<gb, tb, smem>>>(
        dptr_Fu, dptr_u, dptr_blk, nblocks, szpdof_uz, ko_sigma);
    bssnrhs3_stock_boundary<pw, nx, pencil_sz>(dptr_Fu, dptr_u, dptr_blk,
                                               nblocks, szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();
}


template <int pw, int nx, int pencil_sz>
GLOBAL_FUNC __launch_bounds__(343, 3) void eval_rhs4(
    DEVICE_REAL* const dptr_Fu, const DEVICE_REAL* const dptr_u,
    const BlockGPU3D* dptr_blk, DEVICE_UINT nblocks,
    BSSN_EVAR_DERIVS* const dptr_deriv_evars, size_t szpdof_uz,
    DEVICE_REAL ko_sigma) {
    // extern __shared__ DEVICE_REAL shared_mem[];

    // DEVICE_REAL *  su  = shared_mem;
    // DEVICE_REAL *  Du  = shared_mem +     pencil_sz * pencil_sz *pencil_sz;
    // DEVICE_REAL *  DDu = shared_mem + 2 * pencil_sz * pencil_sz *pencil_sz;
    __shared__ DEVICE_REAL su[pencil_sz * pencil_sz * pencil_sz];
    __shared__ DEVICE_REAL Du[pencil_sz * pencil_sz * pencil_sz];

    const DEVICE_UINT sz_per_dof             = szpdof_uz;
    const DEVICE_UINT lambda[4]              = {1, 1, 1, 1};
    const DEVICE_REAL lambda_f[2]            = {1.0, 0.0};
    const DEVICE_INT Z_ID                    = GPUDevice::block_id_x();

    BlockGPU3D c_blk                         = dptr_blk[Z_ID];
    const BlockGPU3D* __restrict__ const blk = &c_blk;
    BSSN_EVAR_DERIVS* const deriv_evars      = dptr_deriv_evars;

    const DEVICE_UINT offset                 = blk->m_offset;
    const DEVICE_UINT actual_nx              = blk->m_sz[0];
    const DEVICE_UINT BLK_SZ                 = nx * nx * nx;
    const int pencils                        = pencil_sz;

    DEVICE_REAL* __restrict__ const a_rhs =
        &dptr_Fu[bssn::VAR::U_ALPHA * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const chi_rhs =
        &dptr_Fu[bssn::VAR::U_CHI * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMGT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMGT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMGT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMGT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMGT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const gt_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMGT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs0 =
        &dptr_Fu[bssn::VAR::U_BETA0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs1 =
        &dptr_Fu[bssn::VAR::U_BETA1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const b_rhs2 =
        &dptr_Fu[bssn::VAR::U_BETA2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs00 =
        &dptr_Fu[bssn::VAR::U_SYMAT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs01 =
        &dptr_Fu[bssn::VAR::U_SYMAT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs02 =
        &dptr_Fu[bssn::VAR::U_SYMAT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs11 =
        &dptr_Fu[bssn::VAR::U_SYMAT3 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs12 =
        &dptr_Fu[bssn::VAR::U_SYMAT4 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const At_rhs22 =
        &dptr_Fu[bssn::VAR::U_SYMAT5 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs0 =
        &dptr_Fu[bssn::VAR::U_GT0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs1 =
        &dptr_Fu[bssn::VAR::U_GT1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const Gt_rhs2 =
        &dptr_Fu[bssn::VAR::U_GT2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs0 =
        &dptr_Fu[bssn::VAR::U_B0 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs1 =
        &dptr_Fu[bssn::VAR::U_B1 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const B_rhs2 =
        &dptr_Fu[bssn::VAR::U_B2 * sz_per_dof + offset];
    DEVICE_REAL* __restrict__ const K_rhs =
        &dptr_Fu[bssn::VAR::U_K * sz_per_dof + offset];

    const DEVICE_REAL* __restrict__ const alpha =
        (dptr_u + bssn::VAR::U_ALPHA * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const chi =
        (dptr_u + bssn::VAR::U_CHI * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const K =
        (dptr_u + bssn::VAR::U_K * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt0 =
        (dptr_u + bssn::VAR::U_SYMGT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt1 =
        (dptr_u + bssn::VAR::U_SYMGT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt2 =
        (dptr_u + bssn::VAR::U_SYMGT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt3 =
        (dptr_u + bssn::VAR::U_SYMGT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt4 =
        (dptr_u + bssn::VAR::U_SYMGT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const gt5 =
        (dptr_u + bssn::VAR::U_SYMGT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta0 =
        (dptr_u + bssn::VAR::U_BETA0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta1 =
        (dptr_u + bssn::VAR::U_BETA1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const beta2 =
        (dptr_u + bssn::VAR::U_BETA2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At0 =
        (dptr_u + bssn::VAR::U_SYMAT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At1 =
        (dptr_u + bssn::VAR::U_SYMAT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At2 =
        (dptr_u + bssn::VAR::U_SYMAT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At3 =
        (dptr_u + bssn::VAR::U_SYMAT3 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At4 =
        (dptr_u + bssn::VAR::U_SYMAT4 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const At5 =
        (dptr_u + bssn::VAR::U_SYMAT5 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt0 =
        (dptr_u + bssn::VAR::U_GT0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt1 =
        (dptr_u + bssn::VAR::U_GT1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const Gt2 =
        (dptr_u + bssn::VAR::U_GT2 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B0 =
        (dptr_u + bssn::VAR::U_B0 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B1 =
        (dptr_u + bssn::VAR::U_B1 * sz_per_dof + offset);
    const DEVICE_REAL* __restrict__ const B2 =
        (dptr_u + bssn::VAR::U_B2 * sz_per_dof + offset);

    const DEVICE_INT i                = GPUDevice::thread_id_x() + pw;
    const DEVICE_INT j                = GPUDevice::thread_id_y() + pw;
    const DEVICE_INT k                = GPUDevice::thread_id_z() + pw;

    const int gidx                    = (k * nx + j) * nx + i;

    const DEVICE_REAL hx              = blk->m_dx[0];

    const DEVICE_REAL x               = blk->m_ptMin[0] + i * hx;
    const DEVICE_REAL y               = blk->m_ptMin[1] + j * blk->m_dx[1];
    const DEVICE_REAL z               = blk->m_ptMin[2] + k * blk->m_dx[2];

    const DEVICE_REAL RIT_ETA_OUTER   = 0.25;
    const DEVICE_REAL RIT_ETA_CENTRAL = 2.0;
    const DEVICE_REAL RIT_ETA_WIDTH   = 40.0;

    const DEVICE_REAL r_coord         = sqrt(x * x + y * y + z * z);

    const DEVICE_REAL w               = r_coord / RIT_ETA_WIDTH;
    const DEVICE_REAL arg             = -w * w * w * w;
    const DEVICE_REAL eta =
        (RIT_ETA_CENTRAL - RIT_ETA_OUTER) * exp(arg) + RIT_ETA_OUTER;

// for(int v=0; v < 24; v++)
// device::__ld_blk_var1__<DEVICE_REAL,pw,nx>   (su , &dptr_u[v*sz_per_dof +
// offset], blk);
#include "../scripts/compute_bssnrhs_evar_derivs4.cuh"
    GPUDevice::sync_threads();
    // device::__blk1_deriv644_x<pw,pencils,pencil_sz>(Du , su, blk);
    // __syncthreads();
    // DEVICE_REAL grad_0_alpha=Du[gidx];
    // __syncthreads();
    // device::__blk1_deriv644_y<pw,pencils,pencil_sz>(Du , su, blk);
    // __syncthreads();
    // DEVICE_REAL grad_1_alpha=Du[gidx];
    // __syncthreads();
    // device::__blk1_deriv644_z<pw,pencils,pencil_sz>(Du , su, blk);
    // __syncthreads();
    // DEVICE_REAL grad_2_alpha=Du[gidx];
    const DEVICE_INT pp   = gidx;
    const DEVICE_INT d_pp = Z_ID * BLK_SZ + gidx;
#include "../scripts/bssneqs_eta_const_standard_gauge4.cpp"
#include "../scripts/bssnrhs_evar_derivs_zid.h"
    GPUDevice::sync_threads();
    if (blk->m_bflag != 0) {
        radiative_bc_pt<pw, nx>(&a_rhs[gidx], alpha[gidx], grad_0_alpha[gidx],
                                grad_1_alpha[gidx], grad_2_alpha[gidx], 1.0,
                                1.0, blk);

        radiative_bc_pt<pw, nx>(&b_rhs0[gidx], beta0[gidx], grad_0_beta0[gidx],
                                grad_1_beta0[gidx], grad_2_beta0[gidx], 1.0,
                                0.0, blk);
        radiative_bc_pt<pw, nx>(&b_rhs1[gidx], beta1[gidx], grad_0_beta1[gidx],
                                grad_1_beta1[gidx], grad_2_beta1[gidx], 1.0,
                                0.0, blk);
        radiative_bc_pt<pw, nx>(&b_rhs2[gidx], beta2[gidx], grad_0_beta2[gidx],
                                grad_1_beta2[gidx], grad_2_beta2[gidx], 1.0,
                                0.0, blk);

        radiative_bc_pt<pw, nx>(&chi_rhs[gidx], chi[gidx], grad_0_chi[gidx],
                                grad_1_chi[gidx], grad_2_chi[gidx], 1.0, 1.0,
                                blk);
        radiative_bc_pt<pw, nx>(&K_rhs[gidx], K[gidx], grad_0_K[gidx],
                                grad_1_K[gidx], grad_2_K[gidx], 1.0, 0.0, blk);

        radiative_bc_pt<pw, nx>(&Gt_rhs0[gidx], Gt0[gidx], grad_0_Gt0[gidx],
                                grad_1_Gt0[gidx], grad_2_Gt0[gidx], 2.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&Gt_rhs1[gidx], Gt1[gidx], grad_0_Gt1[gidx],
                                grad_1_Gt1[gidx], grad_2_Gt1[gidx], 2.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&Gt_rhs2[gidx], Gt2[gidx], grad_0_Gt2[gidx],
                                grad_1_Gt2[gidx], grad_2_Gt2[gidx], 2.0, 0.0,
                                blk);

        radiative_bc_pt<pw, nx>(&B_rhs0[gidx], B0[gidx], grad_0_B0[gidx],
                                grad_1_B0[gidx], grad_2_B0[gidx], 1.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&B_rhs1[gidx], B1[gidx], grad_0_B1[gidx],
                                grad_1_B1[gidx], grad_2_B1[gidx], 1.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&B_rhs2[gidx], B2[gidx], grad_0_B2[gidx],
                                grad_1_B2[gidx], grad_2_B2[gidx], 1.0, 0.0,
                                blk);

        radiative_bc_pt<pw, nx>(&At_rhs00[gidx], At0[gidx], grad_0_At0[gidx],
                                grad_1_At0[gidx], grad_2_At0[gidx], 2.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&At_rhs01[gidx], At1[gidx], grad_0_At1[gidx],
                                grad_1_At1[gidx], grad_2_At1[gidx], 2.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&At_rhs02[gidx], At2[gidx], grad_0_At2[gidx],
                                grad_1_At2[gidx], grad_2_At2[gidx], 2.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&At_rhs11[gidx], At3[gidx], grad_0_At3[gidx],
                                grad_1_At3[gidx], grad_2_At3[gidx], 2.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&At_rhs12[gidx], At4[gidx], grad_0_At4[gidx],
                                grad_1_At4[gidx], grad_2_At4[gidx], 2.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&At_rhs22[gidx], At5[gidx], grad_0_At5[gidx],
                                grad_1_At5[gidx], grad_2_At5[gidx], 2.0, 0.0,
                                blk);

        radiative_bc_pt<pw, nx>(&gt_rhs00[gidx], gt0[gidx], grad_0_gt0[gidx],
                                grad_1_gt0[gidx], grad_2_gt0[gidx], 1.0, 1.0,
                                blk);
        radiative_bc_pt<pw, nx>(&gt_rhs01[gidx], gt1[gidx], grad_0_gt1[gidx],
                                grad_1_gt1[gidx], grad_2_gt1[gidx], 1.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&gt_rhs02[gidx], gt2[gidx], grad_0_gt2[gidx],
                                grad_1_gt2[gidx], grad_2_gt2[gidx], 1.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&gt_rhs11[gidx], gt3[gidx], grad_0_gt3[gidx],
                                grad_1_gt3[gidx], grad_2_gt3[gidx], 1.0, 1.0,
                                blk);
        radiative_bc_pt<pw, nx>(&gt_rhs12[gidx], gt4[gidx], grad_0_gt4[gidx],
                                grad_1_gt4[gidx], grad_2_gt4[gidx], 1.0, 0.0,
                                blk);
        radiative_bc_pt<pw, nx>(&gt_rhs22[gidx], gt5[gidx], grad_0_gt5[gidx],
                                grad_1_gt5[gidx], grad_2_gt5[gidx], 1.0, 1.0,
                                blk);
    }

    GPUDevice::sync_threads();
    const DEVICE_REAL sigma = ko_sigma;
    a_rhs[pp] += sigma * (kograd_0_alpha[gidx] + kograd_1_alpha[gidx] +
                          kograd_2_alpha[gidx]);
    b_rhs0[pp] += sigma * (kograd_0_beta0[gidx] + kograd_1_beta0[gidx] +
                           kograd_2_beta0[gidx]);
    b_rhs1[pp] += sigma * (kograd_0_beta1[gidx] + kograd_1_beta1[gidx] +
                           kograd_2_beta1[gidx]);
    b_rhs2[pp] += sigma * (kograd_0_beta2[gidx] + kograd_1_beta2[gidx] +
                           kograd_2_beta2[gidx]);

    gt_rhs00[pp] +=
        sigma * (kograd_0_gt0[gidx] + kograd_1_gt0[gidx] + kograd_2_gt0[gidx]);
    gt_rhs01[pp] +=
        sigma * (kograd_0_gt1[gidx] + kograd_1_gt1[gidx] + kograd_2_gt1[gidx]);
    gt_rhs02[pp] +=
        sigma * (kograd_0_gt2[gidx] + kograd_1_gt2[gidx] + kograd_2_gt2[gidx]);
    gt_rhs11[pp] +=
        sigma * (kograd_0_gt3[gidx] + kograd_1_gt3[gidx] + kograd_2_gt3[gidx]);
    gt_rhs12[pp] +=
        sigma * (kograd_0_gt4[gidx] + kograd_1_gt4[gidx] + kograd_2_gt4[gidx]);
    gt_rhs22[pp] +=
        sigma * (kograd_0_gt5[gidx] + kograd_1_gt5[gidx] + kograd_2_gt5[gidx]);

    chi_rhs[pp] +=
        sigma * (kograd_0_chi[gidx] + kograd_1_chi[gidx] + kograd_2_chi[gidx]);

    At_rhs00[pp] +=
        sigma * (kograd_0_At0[gidx] + kograd_1_At0[gidx] + kograd_2_At0[gidx]);
    At_rhs01[pp] +=
        sigma * (kograd_0_At1[gidx] + kograd_1_At1[gidx] + kograd_2_At1[gidx]);
    At_rhs02[pp] +=
        sigma * (kograd_0_At2[gidx] + kograd_1_At2[gidx] + kograd_2_At2[gidx]);
    At_rhs11[pp] +=
        sigma * (kograd_0_At3[gidx] + kograd_1_At3[gidx] + kograd_2_At3[gidx]);
    At_rhs12[pp] +=
        sigma * (kograd_0_At4[gidx] + kograd_1_At4[gidx] + kograd_2_At4[gidx]);
    At_rhs22[pp] +=
        sigma * (kograd_0_At5[gidx] + kograd_1_At5[gidx] + kograd_2_At5[gidx]);

    K_rhs[pp] +=
        sigma * (kograd_0_K[gidx] + kograd_1_K[gidx] + kograd_2_K[gidx]);

    Gt_rhs0[pp] +=
        sigma * (kograd_0_Gt0[gidx] + kograd_1_Gt0[gidx] + kograd_2_Gt0[gidx]);
    Gt_rhs1[pp] +=
        sigma * (kograd_0_Gt1[gidx] + kograd_1_Gt1[gidx] + kograd_2_Gt1[gidx]);
    Gt_rhs2[pp] +=
        sigma * (kograd_0_Gt2[gidx] + kograd_1_Gt2[gidx] + kograd_2_Gt2[gidx]);

    B_rhs0[pp] +=
        sigma * (kograd_0_B0[gidx] + kograd_1_B0[gidx] + kograd_2_B0[gidx]);
    B_rhs1[pp] +=
        sigma * (kograd_0_B1[gidx] + kograd_1_B1[gidx] + kograd_2_B1[gidx]);
    B_rhs2[pp] +=
        sigma * (kograd_0_B2[gidx] + kograd_1_B2[gidx] + kograd_2_B2[gidx]);

    return;
}

template <int pw, int pencil_sz, int nx, int BATCHED_GRAIN_SZ, int nstreams>
HOST_FUNC void bssnrhs4(DEVICE_REAL* const dptr_Fu,
                        const DEVICE_REAL* const dptr_u, BlockGPU3D* dptr_blk,
                        DEVICE_UINT nblocks,
                        BSSN_EVAR_DERIVS* const dptr_deriv_evars,
                        size_t szpdof_uz, DEVICE_REAL ko_sigma) {
    dim3 gb = dim3(nblocks, 1, 1);
    dim3 tb = dim3(7, 7, 7);

    eval_rhs4<pw, nx, pencil_sz><<<gb, tb, 0>>>(dptr_Fu, dptr_u, dptr_blk,
                                                nblocks, dptr_deriv_evars,
                                                szpdof_uz, ko_sigma);
    cudaDeviceSynchronize();

    return;
}

}  // namespace device
