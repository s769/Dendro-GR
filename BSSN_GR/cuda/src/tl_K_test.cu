/**
 * Tier 3.1 — K TL vs FD (device PT), product/KO piece of K_rhs
 * (laplacian(α) held fixed / omitted for δK, δα gates).
 *
 *   K_rhs ⊃ (β·∇)K + α (K²/3 + At²) + σ KO(K)
 * δK: J = (β·∇)v + α (2K/3) v + σ KO(v)
 * δα: J = v_α (K²/3 + At²)
 *
 * Usage: tl_K_test [eps]
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
constexpr double KO_SIGMA = 0.4;

__global__ void k_K_rhs(const double* K, const double* alpha,
                        const double* At2, const double* beta0,
                        const double* beta1, const double* beta2, double* rhs) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = K[t];
  __syncthreads();
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  rhs[p] = beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2 +
           alpha[p] * (K[p] * K[p] / 3.0 + At2[p]) +
           KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_K_tl_dK(const double* v, const double* K, const double* alpha,
                          const double* beta0, const double* beta1,
                          const double* beta2, double* J) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = v[t];
  __syncthreads();
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  J[p] = beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2 +
         alpha[p] * (2.0 * K[p] / 3.0) * v[p] + KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_K_tl_dalpha(const double* K, const double* At2,
                              const double* va, double* J) {
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  J[p] = va[p] * (K[p] * K[p] / 3.0 + At2[p]);
}

int compare_tl_fd(const char* tag, const std::vector<double>& rhsp,
                  const std::vector<double>& rhsm, const std::vector<double>& tl,
                  double eps) {
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
      "tl_K_test %s eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n", tag,
      eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;
  std::vector<double> h_K(N), h_a(N), h_At2(N), h_b0(N), h_b1(N), h_b2(N),
      h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_K[p]   = 0.1 * std::cos(4 * x + 3 * y);
        h_a[p]   = 0.9 + 0.04 * std::sin(5 * z);
        h_At2[p] = 0.01 + 0.002 * std::cos(3 * x);
        h_b0[p]  = 0.03 * y;
        h_b1[p]  = -0.03 * x;
        h_b2[p]  = 0.02 * z;
        h_v[p]   = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_K, *d_a, *d_At2, *d_b0, *d_b1, *d_b2, *d_v, *d_rhsp, *d_rhsm, *d_tl;
  for (auto** pp :
       {&d_K, &d_a, &d_At2, &d_b0, &d_b1, &d_b2, &d_v, &d_rhsp, &d_rhsm, &d_tl})
    cudaMalloc(pp, N * sizeof(double));

  cudaMemcpy(d_K, h_K.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_At2, h_At2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);
  std::vector<double> rhsp(N), rhsm(N), tl(N);
  int all_pass = 1;

  auto fd = [&](std::vector<double>& hf, double* df, auto tl_launch,
                const char* tag) {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = hf[p] + eps * h_v[p];
      hm[p] = hf[p] - eps * h_v[p];
    }
    cudaMemcpy(df, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_K_rhs<<<1, tb>>>(d_K, d_a, d_At2, d_b0, d_b1, d_b2, d_rhsp);
    cudaMemcpy(df, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_K_rhs<<<1, tb>>>(d_K, d_a, d_At2, d_b0, d_b1, d_b2, d_rhsm);
    cudaMemcpy(df, hf.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    tl_launch();
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd(tag, rhsp, rhsm, tl, eps);
  };

  fd(h_K, d_K,
     [&] { k_K_tl_dK<<<1, tb>>>(d_v, d_K, d_a, d_b0, d_b1, d_b2, d_tl); },
     "dK");
  fd(h_a, d_a, [&] { k_K_tl_dalpha<<<1, tb>>>(d_K, d_At2, d_v, d_tl); },
     "dalpha");

  std::printf("GATE %s\n", all_pass ? "PASS" : "FAIL");
  for (auto* p : {d_K, d_a, d_At2, d_b0, d_b1, d_b2, d_v, d_rhsp, d_rhsm, d_tl})
    cudaFree(p);
  return all_pass ? 0 : 2;
}
