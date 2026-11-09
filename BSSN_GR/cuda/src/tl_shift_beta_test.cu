/**
 * Tier 3.1 #9 — interior shift β⁰ TL vs FD (device PT).
 *
 * Production (gauge6, λf=[1,0], λ=[1,1,1,1]):
 *   b_rhs0 = B0*(3/4) + λ1 (β·∇)β0 + σ_KO KO(β0)
 * TL in direction v=δβ0 (B0, β1, β2 fixed):
 *   Jb0 = λ1 ( v ∂0β0 + β0 ∂0v + β1 ∂1v + β2 ∂2v ) + σ_KO KO(v)
 *
 * Usage: tl_shift_beta_test [eps]
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "device.h"

namespace device {
#include "derivs_point.cuh"
}

namespace {

constexpr int NX = 13;
constexpr int PW = 3;
constexpr int N  = NX * NX * NX;
constexpr double DX = 1e-2 / (NX - 1);
constexpr double LAMBDA1 = 1.0;
constexpr double KO_SIGMA = 0.4;
constexpr double THREE_FOURTHS = 0.75;

__global__ void k_beta0_rhs(const double* beta0, const double* beta1,
                            const double* beta2, const double* B0,
                            double* b_rhs0) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = beta0[t];
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
  b_rhs0[p] = B0[p] * THREE_FOURTHS +
              LAMBDA1 * (beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2) +
              KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_beta0_tl(const double* v, const double* beta0,
                           const double* beta1, const double* beta2,
                           const double* g0_beta0, double* Jb) {
  // g0_beta0 = ∂0 β0 on background (precomputed) for the v*∂0β0 term
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = v[t];
  __syncthreads();

  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;

  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double gv0, gv1, gv2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &gv0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &gv1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &gv2, &dzz, &k2);
  Jb[p] = LAMBDA1 * (v[p] * g0_beta0[p] + beta0[p] * gv0 + beta1[p] * gv1 +
                     beta2[p] * gv2) +
          KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_grad0_beta0(const double* beta0, double* g0) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = beta0[t];
  __syncthreads();
  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  double gx, dxx, kx;
  device::__pt_fused_x3<NX>(su, i, j, k, 1.0 / DX, &gx, &dxx, &kx);
  g0[p] = gx;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;

  std::vector<double> h_b0(N), h_b1(N), h_b2(N), h_B0(N), h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_b0[p] = 0.03 * y + 0.01 * std::sin(12 * x);
        h_b1[p] = -0.03 * x;
        h_b2[p] = 0.02 * z;
        h_B0[p] = 0.05 * std::cos(8 * x + 4 * z);
        h_v[p]  = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_b0, *d_b1, *d_b2, *d_B0, *d_v, *d_g0;
  double *d_rhs0, *d_rhsp, *d_rhsm, *d_tl;
  cudaMalloc(&d_b0, N * sizeof(double));
  cudaMalloc(&d_b1, N * sizeof(double));
  cudaMalloc(&d_b2, N * sizeof(double));
  cudaMalloc(&d_B0, N * sizeof(double));
  cudaMalloc(&d_v, N * sizeof(double));
  cudaMalloc(&d_g0, N * sizeof(double));
  cudaMalloc(&d_rhs0, N * sizeof(double));
  cudaMalloc(&d_rhsp, N * sizeof(double));
  cudaMalloc(&d_rhsm, N * sizeof(double));
  cudaMalloc(&d_tl, N * sizeof(double));

  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B0, h_B0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);
  k_grad0_beta0<<<1, tb>>>(d_b0, d_g0);
  k_beta0_rhs<<<1, tb>>>(d_b0, d_b1, d_b2, d_B0, d_rhs0);

  std::vector<double> h_bp(N), h_bm(N);
  for (int p = 0; p < N; ++p) {
    h_bp[p] = h_b0[p] + eps * h_v[p];
    h_bm[p] = h_b0[p] - eps * h_v[p];
  }
  cudaMemcpy(d_b0, h_bp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_beta0_rhs<<<1, tb>>>(d_b0, d_b1, d_b2, d_B0, d_rhsp);
  cudaMemcpy(d_b0, h_bm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_beta0_rhs<<<1, tb>>>(d_b0, d_b1, d_b2, d_B0, d_rhsm);
  // restore background β0 for TL (uses background β0 in product rule)
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_beta0_tl<<<1, tb>>>(d_v, d_b0, d_b1, d_b2, d_g0, d_tl);
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

  std::printf(
      "tl_shift_beta_test eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n",
      eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE %s\n", pass ? "PASS" : "FAIL");

  cudaFree(d_b0);
  cudaFree(d_b1);
  cudaFree(d_b2);
  cudaFree(d_B0);
  cudaFree(d_v);
  cudaFree(d_g0);
  cudaFree(d_rhs0);
  cudaFree(d_rhsp);
  cudaFree(d_rhsm);
  cudaFree(d_tl);
  return pass ? 0 : 2;
}
