/**
 * Tier 3.1 #2 — device PT-stencil lapse TL vs FD.
 *
 * Same operator as tl_gauge_fd_test, but ∇/KO from device::__pt_fused_*3
 * (Dendro point stencils). Interior points only (pw=3).
 *
 * Usage: tl_gauge_device_test [eps]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device.h"

// derivs_point.cuh is written to be included inside namespace device.
namespace device {
#include "derivs_point.cuh"
}

namespace {

constexpr int NX = 13;
constexpr int PW = 3;
constexpr int N  = NX * NX * NX;
constexpr double DX = 1e-2 / (NX - 1);
constexpr double LAMBDA0 = 1.0;
constexpr double KO_SIGMA = 0.4;

__global__ void k_lapse_rhs(const double* alpha, const double* K,
                            const double* b0, const double* b1, const double* b2,
                            double* a_rhs) {
  __shared__ double su[N];
  const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = alpha[t];
  __syncthreads();

  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;

  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  a_rhs[p] = -2.0 * K[p] * alpha[p] +
             LAMBDA0 * (b0[p] * g0 + b1[p] * g1 + b2[p] * g2) +
             KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_lapse_tl(const double* v, const double* K, const double* b0,
                           const double* b1, const double* b2, double* Ja) {
  __shared__ double su[N];
  const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = v[t];
  __syncthreads();

  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;

  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  Ja[p] = -2.0 * K[p] * v[p] +
          LAMBDA0 * (b0[p] * g0 + b1[p] * g1 + b2[p] * g2) +
          KO_SIGMA * (k0 + k1 + k2);
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;

  std::vector<double> h_alpha(N), h_K(N), h_b0(N), h_b1(N), h_b2(N), h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_alpha[p] = 1.0 + 0.01 * std::sin(20 * x) * std::cos(15 * y);
        h_K[p]     = 0.1 * std::cos(10 * x + 5 * z);
        h_b0[p]    = 0.02 * y;
        h_b1[p]    = -0.02 * x;
        h_b2[p]    = 0.01 * z;
        h_v[p]     = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_alpha, *d_K, *d_b0, *d_b1, *d_b2, *d_v;
  double *d_rhs0, *d_rhsp, *d_rhsm, *d_tl;
  cudaMalloc(&d_alpha, N * sizeof(double));
  cudaMalloc(&d_K, N * sizeof(double));
  cudaMalloc(&d_b0, N * sizeof(double));
  cudaMalloc(&d_b1, N * sizeof(double));
  cudaMalloc(&d_b2, N * sizeof(double));
  cudaMalloc(&d_v, N * sizeof(double));
  cudaMalloc(&d_rhs0, N * sizeof(double));
  cudaMalloc(&d_rhsp, N * sizeof(double));
  cudaMalloc(&d_rhsm, N * sizeof(double));
  cudaMalloc(&d_tl, N * sizeof(double));

  cudaMemcpy(d_alpha, h_alpha.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);  // 7x7x7
  k_lapse_rhs<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, d_rhs0);

  std::vector<double> h_ap(N), h_am(N);
  for (int p = 0; p < N; ++p) {
    h_ap[p] = h_alpha[p] + eps * h_v[p];
    h_am[p] = h_alpha[p] - eps * h_v[p];
  }
  cudaMemcpy(d_alpha, h_ap.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_lapse_rhs<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, d_rhsp);
  cudaMemcpy(d_alpha, h_am.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_lapse_rhs<<<1, tb>>>(d_alpha, d_K, d_b0, d_b1, d_b2, d_rhsm);
  k_lapse_tl<<<1, tb>>>(d_v, d_K, d_b0, d_b1, d_b2, d_tl);
  cudaDeviceSynchronize();

  std::vector<double> rhsp(N), rhsm(N), tl(N);
  cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);

  double max_abs = 0.0, max_rel = 0.0, max_fd = 0.0;
  int worst = 0;
  for (int k = PW; k < NX - PW; ++k)
    for (int j = PW; j < NX - PW; ++j)
      for (int i = PW; i < NX - PW; ++i) {
        const int p = i + NX * (j + NX * k);
        const double fd = (rhsp[p] - rhsm[p]) / (2.0 * eps);
        const double diff = std::abs(tl[p] - fd);
        const double rel = diff / std::max(1e-16, std::abs(fd));
        if (diff > max_abs) {
          max_abs = diff;
          max_rel = rel;
          max_fd  = fd;
          worst   = p;
        }
      }

  std::printf("tl_gauge_device_test eps=%.3e max|TL-FD|=%.3e rel=%.3e "
              "(fd=%.3e @p=%d)\n",
              eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE %s\n", pass ? "PASS" : "FAIL");

  cudaFree(d_alpha);
  cudaFree(d_K);
  cudaFree(d_b0);
  cudaFree(d_b1);
  cudaFree(d_b2);
  cudaFree(d_v);
  cudaFree(d_rhs0);
  cudaFree(d_rhsp);
  cudaFree(d_rhsm);
  cudaFree(d_tl);
  return pass ? 0 : 2;
}
