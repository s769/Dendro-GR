/**
 * Tier 3.1 — toy Ricci×α coupling TL vs FD (flat linearized R00).
 *
 *   R00 ≈ −½ Δ gt00   (δg-only, ignore trace/Christoffel)
 *   At-like piece:  rhs = −α R00
 *
 * δgt0: J = −α (−½ lap v) = ½ α lap(v)
 * δα:   J = −v_α R00
 *
 * Usage: tl_Ricci_test [eps]
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

__global__ void k_rhs(const double* alpha, const double* R00, double* rhs) {
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  rhs[p] = -alpha[p] * R00[p];
}

__global__ void k_R_from_gt(const double* gt0, double* R00) {
  // R00 = -0.5 * lap(gt0)
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = gt0[t];
  __syncthreads();
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  R00[p] = -0.5 * (dxx + dyy + dzz);
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
      "tl_Ricci_test %s eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n",
      tag, eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;
  std::vector<double> h_gt(N), h_a(N), h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_gt[p] = 1.0 + 0.05 * std::cos(4 * x + 2 * z);
        h_a[p]  = 0.9 + 0.03 * std::sin(3 * y);
        h_v[p]  = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_gt, *d_a, *d_v, *d_R, *d_rhsp, *d_rhsm, *d_tl, *d_tmp;
  for (auto** pp : {&d_gt, &d_a, &d_v, &d_R, &d_rhsp, &d_rhsm, &d_tl, &d_tmp})
    cudaMalloc(pp, N * sizeof(double));
  cudaMemcpy(d_gt, h_gt.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);
  std::vector<double> rhsp(N), rhsm(N), tl(N);
  int all_pass = 1;

  auto run_rhs = [&](double* gt_ptr, double* out) {
    k_R_from_gt<<<1, tb>>>(gt_ptr, d_R);
    k_rhs<<<1, tb>>>(d_a, d_R, out);
  };

  // δgt0
  {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = h_gt[p] + eps * h_v[p];
      hm[p] = h_gt[p] - eps * h_v[p];
    }
    cudaMemcpy(d_gt, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    run_rhs(d_gt, d_rhsp);
    cudaMemcpy(d_gt, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    run_rhs(d_gt, d_rhsm);
    cudaMemcpy(d_gt, h_gt.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    // J = 0.5 * alpha * lap(v)
    k_lap<<<1, tb>>>(d_v, d_tmp);
    // tl = 0.5 * alpha * tmp
    cudaMemcpy(tl.data(), d_tmp, N * sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<double> ah(N);
    cudaMemcpy(ah.data(), d_a, N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int p = 0; p < N; ++p) tl[p] = 0.5 * ah[p] * tl[p];
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd("dgt0", rhsp, rhsm, tl, eps);
  }

  // δα
  {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = h_a[p] + eps * h_v[p];
      hm[p] = h_a[p] - eps * h_v[p];
    }
    k_R_from_gt<<<1, tb>>>(d_gt, d_R);
    cudaMemcpy(d_a, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_rhs<<<1, tb>>>(d_a, d_R, d_rhsp);
    cudaMemcpy(d_a, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_rhs<<<1, tb>>>(d_a, d_R, d_rhsm);
    cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    // J = -v * R00
    std::vector<double> Rh(N);
    cudaMemcpy(Rh.data(), d_R, N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int p = 0; p < N; ++p) tl[p] = -h_v[p] * Rh[p];
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd("dalpha", rhsp, rhsm, tl, eps);
  }

  std::printf("GATE %s\n", all_pass ? "PASS" : "FAIL");
  for (auto* p : {d_gt, d_a, d_v, d_R, d_rhsp, d_rhsm, d_tl, d_tmp}) cudaFree(p);
  return all_pass ? 0 : 2;
}
