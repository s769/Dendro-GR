/**
 * Tier 3.1 — lap(α) TL vs FD (flat igt=δ, C=0).
 * Piece held out of tl_K_test / used in K_rhs and At Ricci coupling.
 *
 *   lap(α) = ∂xx α + ∂yy α + ∂zz α
 * δα: J = lap(v)
 *
 * Usage: tl_lap_alpha_test [eps]
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

__global__ void k_lap(const double* f, double* out) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = f[t];
  __syncthreads();
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  out[p] = dxx + dyy + dzz;
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
      "tl_lap_alpha_test %s eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n",
      tag, eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;
  std::vector<double> h_a(N), h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_a[p] = 0.9 + 0.04 * std::sin(5 * x) * std::cos(3 * y + z);
        h_v[p] = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_a, *d_v, *d_rhsp, *d_rhsm, *d_tl;
  for (auto** pp : {&d_a, &d_v, &d_rhsp, &d_rhsm, &d_tl})
    cudaMalloc(pp, N * sizeof(double));
  cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);
  std::vector<double> rhsp(N), rhsm(N), tl(N), hp(N), hm(N);
  for (int p = 0; p < N; ++p) {
    hp[p] = h_a[p] + eps * h_v[p];
    hm[p] = h_a[p] - eps * h_v[p];
  }
  cudaMemcpy(d_a, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_lap<<<1, tb>>>(d_a, d_rhsp);
  cudaMemcpy(d_a, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_lap<<<1, tb>>>(d_a, d_rhsm);
  k_lap<<<1, tb>>>(d_v, d_tl);
  cudaDeviceSynchronize();
  cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
  const int pass = compare_tl_fd("dalpha", rhsp, rhsm, tl, eps);
  std::printf("GATE %s\n", pass ? "PASS" : "FAIL");
  for (auto* p : {d_a, d_v, d_rhsp, d_rhsm, d_tl}) cudaFree(p);
  return pass ? 0 : 2;
}
