/**
 * @brief CPU-GPU rhs comparison.
 */

// TODO: figure out proper ordering for these includes
// clang-format off
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device.h"
#include "point.h"
#include "derivs_cu.cuh"
#include "derivs.h"
#include "block_gpu.h"
#include "profiler.h"
#include "grDef.h"
#include "grUtils.h"
#include "rhs.h"
#include "test_utils.h"
#include "bssn_kernels.cuh"
// clang-format on

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <nblocks> <nReps> <bflag>"
                  << std::endl;
        return 1;
    }
    const unsigned int nblocks   = atoi(argv[1]);
    const unsigned int nReps     = atoi(argv[2]);
    const unsigned int bflag     = atoi(argv[3]);

    const unsigned int blk_sz_1d = 13;
    const unsigned int pencils   = 13;
    const Point grid_min(-0.5, -0.5, -0.5);
    const unsigned int sz[3] = {blk_sz_1d, blk_sz_1d, blk_sz_1d};
    const Point dx(1.0e-2 / sz[0], 1.0e-2 / sz[1], 1.0e-2 / sz[2]);
    const unsigned int dof = bssn::BSSN_NUM_VARS;

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    printf("number of cuda devices: %d\n", devicesCount);

    cudaSetDevice(0);
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    std::vector<BlockGPU3D> blk;
    blk.resize(nblocks);

    blk[0].m_offset = 0;
    for (unsigned int i = 0; i < blk.size(); i++) {
        blk[i].m_sz[0]         = sz[0];
        blk[i].m_sz[1]         = sz[1];
        blk[i].m_sz[2]         = sz[2];
        blk[i].m_aligned_sz[0] = sz[0];
        blk[i].m_aligned_sz[1] = sz[1];
        blk[i].m_aligned_sz[2] = sz[2];
        blk[i].m_dx[0]         = dx.x() / (i % 3 + 1);
        blk[i].m_dx[1]         = dx.y() / (i % 3 + 1);
        blk[i].m_dx[2]         = dx.z() / (i % 3 + 1);

        blk[i].m_ptMin[0]      = grid_min.x();  //+ 0.12* (i%10);
        blk[i].m_ptMin[1]      = grid_min.y();  //+ 0.12* (i%10);
        blk[i].m_ptMin[2]      = grid_min.z();  //+ 0.12* (i%10);
        blk[i].m_bflag         = bflag;

        if (i > 0)
            blk[i].m_offset =
                blk[i - 1].m_offset + blk[i - 1].m_aligned_sz[0] *
                                          blk[i - 1].m_aligned_sz[1] *
                                          blk[i - 1].m_aligned_sz[2];
    }
    const double ko_sigma = 4e-1;
    const unsigned int szp_dof =
        (blk.back().m_offset + blk.back().m_aligned_sz[0] *
                                   blk.back().m_aligned_sz[1] *
                                   blk.back().m_aligned_sz[2]);
    const unsigned int blk_vec_sz =
        (blk.back().m_offset + blk.back().m_aligned_sz[0] *
                                   blk.back().m_aligned_sz[1] *
                                   blk.back().m_aligned_sz[2]) *
        dof;
    double* u            = create_block<double>(blk.data(), blk.size(), dof);
    double* Fu_cpu       = new double[blk_vec_sz];
    double* Fu_gpu       = new double[blk_vec_sz];

    double* dptr_u       = GPUDevice::device_malloc<double>(blk_vec_sz);
    double* dptr_Fu      = GPUDevice::device_malloc<double>(blk_vec_sz);

    BlockGPU3D* dptr_blk = GPUDevice::device_malloc<BlockGPU3D>(nblocks);
    GPUDevice::host_to_device(blk.data(), dptr_blk, nblocks);
    GPUDevice::host_to_device(u, dptr_u, blk_vec_sz);

    profiler_t t_cpu;
    profiler_t t_gpu;

    const char* omp_env           = std::getenv("OMP_NUM_THREADS");
    const unsigned int thread_qty = omp_env ? std::max(atoi(omp_env), 1) : 1;

    const unsigned int num_bssn_derivs = 210;
    const unsigned int blk_batch_size  = 1;
    std::cout << "OMP_NUM_THREADS=" << thread_qty << std::endl;
    std::cout << "nblocks=" << nblocks << std::endl;

    // make sure we set the appropriate derivatives, padding width is 3
    set_appropriate_derivs(3);

    double* deriv_workspace =
        (double*)malloc(sizeof(double) * thread_qty * num_bssn_derivs *
                        blk_sz_1d * blk_sz_1d * blk_sz_1d);
    bssnrhs_cpu<3, pencils, blk_sz_1d>(Fu_cpu, u, blk.data(), blk.size(),
                                       deriv_workspace, szp_dof, ko_sigma);

#pragma nounroll
    for (unsigned int i = 0; i < nReps; i++) {
        t_cpu.start();
        bssnrhs_cpu<3, pencils, blk_sz_1d>(Fu_cpu, u, blk.data(), blk.size(),
                                           deriv_workspace, szp_dof, ko_sigma);
        t_cpu.stop();
        // printf("iter = %d time = %f snap= %f\n",i, (double)t_cpu.seconds,
        // (double)t_cpu.snap);
        //  Fu_cpu[0]+=0;
        //  u[0]+=1;
    }

    free(deriv_workspace);

    // Print device and precision
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nDevice Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    const unsigned int nstreams      = 1;
    BSSN_EVAR_DERIVS* deriv_evars_ws = new BSSN_EVAR_DERIVS[nstreams];
    BSSN_EVAR_DERIVS* dptr_deriv_evars_ws =
        GPUDevice::device_malloc<BSSN_EVAR_DERIVS>(nstreams);
    const unsigned int BLK_SZ      = blk_sz_1d * blk_sz_1d * blk_sz_1d;
    const unsigned int sz_p_stream = num_bssn_derivs * blk_batch_size * BLK_SZ;
    DEVICE_REAL* deriv_base_ws =
        GPUDevice::device_malloc<DEVICE_REAL>(nstreams * sz_p_stream);

    for (unsigned int i = 0; i < nstreams; i++) {
        const unsigned int BATCHED_BLOCKS_SZ = blk_batch_size;
        DEVICE_REAL* deriv_base              = deriv_base_ws + i * sz_p_stream;
        BSSN_EVAR_DERIVS* deriv_evars        = deriv_evars_ws + i;
#include "../scripts/bssnrhs_memalloc.cuh"
    }
    GPUDevice::host_to_device(deriv_evars_ws, dptr_deriv_evars_ws, nstreams);
    // device::bssnrhs_gpu<3,pencils,blk_sz_1d, blk_batch_size,
    // blk_sz_1d>(dptr_Fu,dptr_u,dptr_blk, blk.size(),dptr_deriv_evars, BLK_SZ,
    // ko_sigma); cudaDeviceSynchronize();
    // cudaFuncSetCacheConfig(&device::bssnrhs<3,pencils,blk_sz_1d, blk_sz_1d>,
    // cudaFuncCachePreferL1); dim3 gb = dim3(blk_batch_size, 1, 1); dim3 tb =
    // dim3(blk_sz_1d, blk_sz_1d, 1); device::bssnrhs<3,pencils,blk_sz_1d,
    // blk_sz_1d><<<gb,tb>>>(dptr_Fu,dptr_u,dptr_blk,
    // blk.size(),dptr_deriv_evars, szp_dof, ko_sigma); cudaDeviceSynchronize();

    // t_gpu.start();
    // for(unsigned int i=0; i < nReps; i++){
    //   device::bssnrhs<3,pencils,blk_sz_1d,
    //   blk_sz_1d><<<gb,tb>>>(dptr_Fu,dptr_u,dptr_blk,
    //   blk.size(),dptr_deriv_evars, szp_dof, ko_sigma);
    //   cudaDeviceSynchronize();
    // }
    // t_gpu.stop();
    int maxbytes = 13 * 13 * 13 * 8 * 3;  // 96 KB
    cudaFuncSetAttribute(device::eval_rhs3<3, blk_sz_1d, blk_sz_1d>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    // cudaFuncSetAttribute(device::eval_rhs4<3,blk_sz_1d, blk_sz_1d>,
    // cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    // cudaFuncSetCacheConfig(&device::eval_rhs3<3,blk_sz_1d, blk_sz_1d>,
    // cudaFuncCachePreferEqual);
    device::bssnrhs3<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>(
        dptr_Fu, dptr_u, dptr_blk, blk.size(), szp_dof, ko_sigma);
    cudaDeviceSynchronize();
    GPUDevice::check_last_error();

    for (unsigned int i = 0; i < nReps; i++) {
        t_gpu.start();
        device::bssnrhs3<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>(
            dptr_Fu, dptr_u, dptr_blk, blk.size(), szp_dof, ko_sigma);
        cudaDeviceSynchronize();
        t_gpu.stop();
    }

    // Optional: workspace-deriv path (eval_rhs4). Needs deriv WS sized to nblocks.
    if (std::getenv("DENDRO_TIME_PT")) {
        profiler_t t_pt;
        device::bssnrhs3_pt<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>(
            dptr_Fu, dptr_u, dptr_blk, blk.size(), szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_pt.start();
            device::bssnrhs3_pt<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                nstreams>(dptr_Fu, dptr_u, dptr_blk, blk.size(),
                                          szp_dof, ko_sigma);
            t_pt.stop();
        }
        printf("pt_gpu_time\t%.4f\n", (double)t_pt.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

    if (std::getenv("DENDRO_TIME_PT_ASYNC")) {
        profiler_t t_async;
        device::bssnrhs3_pt_async<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                  nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                            blk.size(), szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_async.start();
            device::bssnrhs3_pt_async<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                      nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                                blk.size(), szp_dof, ko_sigma);
            t_async.stop();
        }
        printf("pt_async_gpu_time\t%.4f\n", (double)t_async.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);

        // Tier 1.3: ensemble K as grid.y on shared block list.
        const char* ens_env = std::getenv("DENDRO_ENSEMBLE_K");
        const unsigned int K = ens_env ? std::max(atoi(ens_env), 1) : 0;
        if (K > 1) {
            double* dptr_uK =
                GPUDevice::device_malloc<double>(blk_vec_sz * K);
            double* dptr_FuK =
                GPUDevice::device_malloc<double>(blk_vec_sz * K);
            for (unsigned int k = 0; k < K; ++k) {
                cudaMemcpy(dptr_uK + k * blk_vec_sz, dptr_u,
                           blk_vec_sz * sizeof(double),
                           cudaMemcpyDeviceToDevice);
            }
            // Serial-K baseline: K independent launches.
            profiler_t t_serial;
            for (unsigned int i = 0; i < nReps; i++) {
                t_serial.start();
                for (unsigned int k = 0; k < K; ++k) {
                    device::bssnrhs3_pt_async<3, blk_sz_1d, blk_sz_1d,
                                              blk_batch_size, nstreams>(
                        dptr_FuK + k * blk_vec_sz, dptr_uK + k * blk_vec_sz,
                        dptr_blk, blk.size(), szp_dof, ko_sigma, 1);
                }
                t_serial.stop();
            }
            // Fused v1: one launch, grid.y = K.
            profiler_t t_ens;
            device::bssnrhs3_pt_async<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                      nstreams>(dptr_FuK, dptr_uK, dptr_blk,
                                                blk.size(), szp_dof, ko_sigma,
                                                K, false);
            GPUDevice::check_last_error();
            for (unsigned int i = 0; i < nReps; i++) {
                t_ens.start();
                device::bssnrhs3_pt_async<3, blk_sz_1d, blk_sz_1d,
                                          blk_batch_size, nstreams>(
                    dptr_FuK, dptr_uK, dptr_blk, blk.size(), szp_dof, ko_sigma,
                    K, false);
                t_ens.stop();
            }
            printf("pt_async_serialK_time\tK=%u\t%.4f\n", K,
                   (double)t_serial.seconds);
            printf("pt_async_ensK_time\tK=%u\t%.4f\n", K,
                   (double)t_ens.seconds);
            printf("pt_async_ensK_speedup\tK=%u\t%.3f\n", K,
                   (double)t_serial.seconds /
                       std::max((double)t_ens.seconds, 1e-12));

            double* Fu_ref = new double[blk_vec_sz];
            double* Fu_slot = new double[blk_vec_sz];
            std::memcpy(Fu_ref, Fu_gpu, blk_vec_sz * sizeof(double));
            unsigned int slot_mismatch = 0;
            for (unsigned int k = 0; k < K; ++k) {
                GPUDevice::device_to_host(Fu_slot, dptr_FuK + k * blk_vec_sz,
                                          blk_vec_sz);
                for (unsigned int i = 0; i < blk_vec_sz; ++i) {
                    if (Fu_slot[i] != Fu_ref[i]) {
                        ++slot_mismatch;
                        break;
                    }
                }
            }
            printf("pt_async_ensK_bitid_slots\tK=%u\tmismatched_slots=%u\n", K,
                   slot_mismatch);

            // Tier 1.3 v2: K members/CTA (geometry shared; loop members).
            profiler_t t_share;
            device::bssnrhs3_pt_async<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                      nstreams>(dptr_FuK, dptr_uK, dptr_blk,
                                                blk.size(), szp_dof, ko_sigma,
                                                K, true);
            GPUDevice::check_last_error();
            for (unsigned int i = 0; i < nReps; i++) {
                t_share.start();
                device::bssnrhs3_pt_async<3, blk_sz_1d, blk_sz_1d,
                                          blk_batch_size, nstreams>(
                    dptr_FuK, dptr_uK, dptr_blk, blk.size(), szp_dof, ko_sigma,
                    K, true);
                t_share.stop();
            }
            printf("pt_async_kshare_time\tK=%u\t%.4f\n", K,
                   (double)t_share.seconds);
            printf("pt_async_kshare_vs_ensK\tK=%u\t%.3f\n", K,
                   (double)t_ens.seconds /
                       std::max((double)t_share.seconds, 1e-12));
            printf("pt_async_kshare_vs_single\tK=%u\t%.3f\n", K,
                   (double)t_share.seconds /
                       std::max((double)t_async.seconds, 1e-12));

            unsigned int kshare_mismatch = 0;
            for (unsigned int k = 0; k < K; ++k) {
                GPUDevice::device_to_host(Fu_slot, dptr_FuK + k * blk_vec_sz,
                                          blk_vec_sz);
                for (unsigned int i = 0; i < blk_vec_sz; ++i) {
                    if (Fu_slot[i] != Fu_ref[i]) {
                        ++kshare_mismatch;
                        break;
                    }
                }
            }
            printf("pt_async_kshare_bitid_slots\tK=%u\tmismatched_slots=%u\n", K,
                   kshare_mismatch);
            delete[] Fu_ref;
            delete[] Fu_slot;
            GPUDevice::device_free(dptr_uK);
            GPUDevice::device_free(dptr_FuK);
        }
    }

    if (std::getenv("DENDRO_TIME_PT_GMEM")) {
        profiler_t t_gmem;
        device::bssnrhs3_pt_gmem<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                 nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                           blk.size(), szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_gmem.start();
            device::bssnrhs3_pt_gmem<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                     nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                               blk.size(), szp_dof, ko_sigma);
            t_gmem.stop();
        }
        printf("pt_gmem_gpu_time\t%.4f\n", (double)t_gmem.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

    if (std::getenv("DENDRO_TIME_PT_F32SU")) {
        profiler_t t_f32su;
        device::bssnrhs3_pt_f32su<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                  nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                            blk.size(), szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_f32su.start();
            device::bssnrhs3_pt_f32su<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                      nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                                blk.size(), szp_dof, ko_sigma);
            t_f32su.stop();
        }
        printf("pt_f32su_gpu_time\t%.4f\n", (double)t_f32su.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

    if (std::getenv("DENDRO_TIME_PT_DB")) {
        profiler_t t_db;
        device::bssnrhs3_pt_db<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                               nstreams>(dptr_Fu, dptr_u, dptr_blk, blk.size(),
                                         szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_db.start();
            device::bssnrhs3_pt_db<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                   nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                             blk.size(), szp_dof, ko_sigma);
            t_db.stop();
        }
        printf("pt_db_gpu_time\t%.4f\n", (double)t_db.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

    if (std::getenv("DENDRO_TIME_PT_KOF32")) {
        profiler_t t_kof;
        device::bssnrhs3_pt_kof32<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                  nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                            blk.size(), szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_kof.start();
            device::bssnrhs3_pt_kof32<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                      nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                                blk.size(), szp_dof, ko_sigma);
            t_kof.stop();
        }
        printf("pt_kof32_gpu_time\t%.4f\n", (double)t_kof.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

    if (std::getenv("DENDRO_TIME_F2D")) {
        profiler_t t_f2d;
        device::bssnrhs3_f2d<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>(
            dptr_Fu, dptr_u, dptr_blk, blk.size(), szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_f2d.start();
            device::bssnrhs3_f2d<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                 nstreams>(dptr_Fu, dptr_u, dptr_blk, blk.size(),
                                           szp_dof, ko_sigma);
            t_f2d.stop();
        }
        printf("f2d_gpu_time\t%.4f\n", (double)t_f2d.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

    if (std::getenv("DENDRO_TIME_F32")) {
        profiler_t t_f32;
        device::bssnrhs3_f32<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>(
            dptr_Fu, dptr_u, dptr_blk, blk.size(), szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_f32.start();
            device::bssnrhs3_f32<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                 nstreams>(dptr_Fu, dptr_u, dptr_blk, blk.size(),
                                           szp_dof, ko_sigma);
            t_f32.stop();
        }
        printf("f32_gpu_time\t%.4f\n", (double)t_f32.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

    if (std::getenv("DENDRO_TIME_F32C")) {
        profiler_t t_f32c;
        device::bssnrhs3_f32c<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                              nstreams>(dptr_Fu, dptr_u, dptr_blk, blk.size(),
                                        szp_dof, ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_f32c.start();
            device::bssnrhs3_f32c<3, blk_sz_1d, blk_sz_1d, blk_batch_size,
                                  nstreams>(dptr_Fu, dptr_u, dptr_blk,
                                            blk.size(), szp_dof, ko_sigma);
            t_f32c.stop();
        }
        printf("f32c_gpu_time\t%.4f\n", (double)t_f32c.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

    if (std::getenv("DENDRO_TIME_RHS2")) {
        profiler_t t_rhs2;
        device::bssnrhs2<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>(
            dptr_Fu, dptr_u, dptr_blk, blk.size(), dptr_deriv_evars_ws, szp_dof,
            ko_sigma);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_rhs2.start();
            device::bssnrhs2<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>(
                dptr_Fu, dptr_u, dptr_blk, blk.size(), dptr_deriv_evars_ws,
                szp_dof, ko_sigma);
            t_rhs2.stop();
        }
        printf("rhs2_gpu_time\t%.4f\n", (double)t_rhs2.seconds);
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
    }

#if DENDRO_HAS_WS_KERNELS
    if (std::getenv("DENDRO_TIME_WS")) {
        const unsigned int nb = static_cast<unsigned int>(blk.size());
        const size_t ws_elems = 210ull * nb * 343ull;
        DEVICE_REAL* grad_ws = GPUDevice::device_malloc<DEVICE_REAL>(ws_elems);
        profiler_t t_ws;
        device::bssnrhs3_ws<3, blk_sz_1d, blk_sz_1d>(
            dptr_Fu, dptr_u, dptr_blk, nb, szp_dof, ko_sigma, grad_ws);
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_ws.start();
            device::bssnrhs3_ws<3, blk_sz_1d, blk_sz_1d>(
                dptr_Fu, dptr_u, dptr_blk, nb, szp_dof, ko_sigma, grad_ws);
            t_ws.stop();
        }
        printf("ws_gpu_time\t%.4f\n", (double)t_ws.seconds);
        // Compare against CPU Fu already computed: reuse Fu_gpu path after
        GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);
        GPUDevice::device_free(grad_ws);
    }
#endif

    if (std::getenv("DENDRO_TIME_RHS4")) {
        const unsigned int nb = static_cast<unsigned int>(blk.size());
        const unsigned int sz_all = num_bssn_derivs * nb * BLK_SZ;
        DEVICE_REAL* deriv_base_all =
            GPUDevice::device_malloc<DEVICE_REAL>(sz_all);
        BSSN_EVAR_DERIVS deriv_evars_all;
        BSSN_EVAR_DERIVS* dptr_deriv_all =
            GPUDevice::device_malloc<BSSN_EVAR_DERIVS>(1);
        {
            const unsigned int BATCHED_BLOCKS_SZ = nb;
            DEVICE_REAL* deriv_base = deriv_base_all;
            BSSN_EVAR_DERIVS* deriv_evars = &deriv_evars_all;
#include "../scripts/bssnrhs_memalloc.cuh"
        }
        GPUDevice::host_to_device(&deriv_evars_all, dptr_deriv_all, 1);
        profiler_t t_rhs4;
        device::bssnrhs4<3, blk_sz_1d, blk_sz_1d, 1, 1>(
            dptr_Fu, dptr_u, dptr_blk, nb, dptr_deriv_all, szp_dof, ko_sigma);
        cudaDeviceSynchronize();
        GPUDevice::check_last_error();
        for (unsigned int i = 0; i < nReps; i++) {
            t_rhs4.start();
            device::bssnrhs4<3, blk_sz_1d, blk_sz_1d, 1, 1>(
                dptr_Fu, dptr_u, dptr_blk, nb, dptr_deriv_all, szp_dof,
                ko_sigma);
            cudaDeviceSynchronize();
            t_rhs4.stop();
        }
        printf("rhs4_gpu_time\t%.4f\n", (double)t_rhs4.seconds);
        GPUDevice::device_free(deriv_base_all);
        GPUDevice::device_free(dptr_deriv_all);
    }

    // device::bssnrhs4<3, blk_sz_1d, blk_sz_1d, blk_batch_size,nstreams>
    // (dptr_Fu,dptr_u,dptr_blk, blk.size(), dptr_deriv_evars_ws , szp_dof,
    // ko_sigma); cudaDeviceSynchronize(); GPUDevice::check_last_error();

    // for(unsigned int i=0; i < nReps; i++){
    //   t_gpu.start();
    //   device::bssnrhs4<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>
    //   (dptr_Fu,dptr_u,dptr_blk, blk.size(),dptr_deriv_evars_ws, szp_dof,
    //   ko_sigma); cudaDeviceSynchronize(); t_gpu.stop();
    // }

    // device::bssnrhs2<3, blk_sz_1d, blk_sz_1d, blk_batch_size,nstreams>
    // (dptr_Fu,dptr_u,dptr_blk, blk.size(), dptr_deriv_evars_ws, szp_dof,
    // ko_sigma); cudaDeviceSynchronize(); GPUDevice::check_last_error();

    // for(unsigned int i=0; i < nReps; i++){
    //   t_gpu.start();
    //   device::bssnrhs2<3, blk_sz_1d, blk_sz_1d, blk_batch_size, nstreams>
    //   (dptr_Fu,dptr_u,dptr_blk, blk.size(), dptr_deriv_evars_ws, szp_dof,
    //   ko_sigma); cudaDeviceSynchronize(); t_gpu.stop();
    // }

    // dim3 gb = dim3(1, blk_sz_1d, blk_batch_size);
    // dim3 tb = dim3(blk_sz_1d, blk_sz_1d, 1);
    // {
    //   const unsigned int nblocks = blk.size();
    //   void* kernelArgs[] = {&dptr_Fu, &dptr_u, &dptr_blk, (void * )&nblocks,
    //   &dptr_deriv_evars, (void *)&BLK_SZ,  (void * )&ko_sigma/* add kernel
    //   args */ };
    //   //device::bssnrhs1<3,pencils,blk_sz_1d, blk_batch_size,
    //   blk_sz_1d><<<gb,tb>>>(dptr_Fu,dptr_u,dptr_blk,
    //   blk.size(),dptr_deriv_evars, BLK_SZ, ko_sigma);
    //   cudaLaunchCooperativeKernel((void*)device::bssnrhs1<3,pencils,blk_sz_1d,
    //   blk_batch_size, blk_sz_1d>, gb, tb, kernelArgs);
    //   GPUDevice::check_last_error();
    //   cudaDeviceSynchronize();
    // }

    // t_gpu.start();
    // for(unsigned int i=0; i < nReps; i++)
    // {
    //   const unsigned int nblocks = blk.size();
    //   void* kernelArgs[] = {&dptr_Fu, &dptr_u, &dptr_blk, (void * )&nblocks,
    //   &dptr_deriv_evars, (void *)&BLK_SZ,  (void * )&ko_sigma/* add kernel
    //   args */ };
    //   //device::bssnrhs1<3,pencils,blk_sz_1d, blk_batch_size,
    //   blk_sz_1d><<<gb,tb>>>(dptr_Fu,dptr_u,dptr_blk,
    //   blk.size(),dptr_deriv_evars, BLK_SZ, ko_sigma);
    //   cudaLaunchCooperativeKernel((void*)device::bssnrhs1<3,pencils,blk_sz_1d,
    //   blk_batch_size, blk_sz_1d>, gb, tb, kernelArgs);
    //   GPUDevice::check_last_error();
    //   cudaDeviceSynchronize();
    // }
    // t_gpu.stop();

    GPUDevice::device_to_host(Fu_gpu, dptr_Fu, blk_vec_sz);

    // Optional: dump Fu for run-to-run bit identity (HANDOFF_NEXT A0).
    if (const char* dump = std::getenv("DENDRO_DUMP_FU")) {
        FILE* f = fopen(dump, "wb");
        if (f) {
            fwrite(&blk_vec_sz, sizeof(blk_vec_sz), 1, f);
            fwrite(Fu_gpu, sizeof(double), blk_vec_sz, f);
            fclose(f);
            printf("wrote_fu_dump\t%s\t%u\n", dump, blk_vec_sz);
        }
    }

    printf("cpu_time\tgpu_time\tspeedup\n");
    printf("%.4f\t%.4f\t%.2f\n", (double)t_cpu.seconds, (double)t_gpu.seconds,
           (double)t_cpu.seconds / (double)t_gpu.seconds);

    GPUDevice::device_free(deriv_base_ws);
    GPUDevice::device_free(dptr_deriv_evars_ws);
    delete[] deriv_evars_ws;
    GPUDevice::device_free(dptr_u);
    GPUDevice::device_free(dptr_Fu);

    const unsigned int PW = 3;
    double l_inf_error[dof];
    for (unsigned int v = 0; v < dof; v++) l_inf_error[v] = 0.0;

    for (unsigned int blk_id = 0; blk_id < blk.size(); blk_id++) {
        const unsigned int offset = blk[blk_id].m_offset;
        const unsigned int a_nx   = blk[blk_id].m_sz[0];
        const unsigned int a_ny   = blk[blk_id].m_sz[1];
        const unsigned int a_nz   = blk[blk_id].m_sz[2];

        const unsigned int nx     = blk[blk_id].m_aligned_sz[0];
        const unsigned int ny     = blk[blk_id].m_aligned_sz[1];
        const unsigned int nz     = blk[blk_id].m_aligned_sz[2];

        const unsigned int blkSz  = nx * ny * nz;

        for (unsigned int k = PW; k < a_nz - PW; k++)
            for (unsigned int j = PW; j < a_ny - PW; j++)
                for (unsigned int i = PW; i < a_nz - PW; i++)
                    for (unsigned int v = 0; v < dof; v++) {
                        const unsigned int pp =
                            v * szp_dof + offset + k * nx * ny + j * nx + i;
                        const double error_metric =
                            fabs((Fu_cpu[pp] - Fu_gpu[pp]) / Fu_cpu[pp]);
                        if (l_inf_error[v] < error_metric) {
                            l_inf_error[v] = error_metric;
                            // if(v== bssn::VAR::U_SYMGT0)
                            //   printf("idx: %d\n ", k*nx*ny + j*nx + i);
                        }
                    }
    }

    for (unsigned int v = 0; v < dof; v++)
        std::cout << "l_inf var " << bssn::BSSN_VAR_NAMES[v] << " = "
                  << l_inf_error[v] << std::endl;

    delete[] u;
    delete[] Fu_cpu;
    delete[] Fu_gpu;

    return 0;
}
