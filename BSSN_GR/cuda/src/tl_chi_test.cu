/**
 * Tier 3.1 — conformal χ TL vs FD (device PT).
 * Matches CodeGen/bssn_tl_gauge.py chi_rhs gates.
 *
 *   chi_rhs = (β·∇)χ − (2/3) χ ∂_i β^i + (2/3) χ α K + σ_KO KO(χ)
 *
 * Gate A δχ:  J = (β·∇)v − (2/3) v ∂iβi + (2/3) v α K + σ KO(v)
 * Gate B δα:  J = (2/3) χ v_α K
 * Gate C δK:  J = (2/3) χ α v_K
 *
 * Usage: tl_chi_test [eps]
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
constexpr double TWO_THIRDS = 2.0 / 3.0;

__global__ void k_grad_dir(const double* f, int dir, double* gout) {
  // dir: 0=x, 1=y, 2=z — one shared tile (fits SM shared limit)
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = f[t];
  __syncthreads();
  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g, d2, ko;
  if (dir == 0)
    device::__pt_fused_x3<NX>(su, i, j, k, inv, &g, &d2, &ko);
  else if (dir == 1)
    device::__pt_fused_y3<NX>(su, i, j, k, inv, &g, &d2, &ko);
  else
    device::__pt_fused_z3<NX>(su, i, j, k, inv, &g, &d2, &ko);
  gout[p] = g;
}

__global__ void k_sum3(const double* a, const double* b, const double* c,
                       double* out) {
  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  out[p] = a[p] + b[p] + c[p];
}

__global__ void k_chi_rhs(const double* chi, const double* alpha, const double* K,
                          const double* beta0, const double* beta1,
                          const double* beta2, const double* divb,
                          double* chi_rhs) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = chi[t];
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
  chi_rhs[p] = beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2 -
               TWO_THIRDS * chi[p] * divb[p] +
               TWO_THIRDS * chi[p] * alpha[p] * K[p] +
               KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_chi_tl_dchi(const double* v, const double* alpha,
                              const double* K, const double* beta0,
                              const double* beta1, const double* beta2,
                              const double* divb, double* J) {
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
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  J[p] = beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2 -
         TWO_THIRDS * v[p] * divb[p] + TWO_THIRDS * v[p] * alpha[p] * K[p] +
         KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_chi_tl_dalpha(const double* chi, const double* v_alpha,
                                const double* K, double* J) {
  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  J[p] = TWO_THIRDS * chi[p] * v_alpha[p] * K[p];
}

__global__ void k_chi_tl_dK(const double* chi, const double* alpha,
                            const double* v_K, double* J) {
  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  J[p] = TWO_THIRDS * chi[p] * alpha[p] * v_K[p];
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
      "tl_chi_test %s eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n", tag,
      eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;

  std::vector<double> h_chi(N), h_a(N), h_K(N), h_b0(N), h_b1(N), h_b2(N),
      h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_chi[p] = 0.8 + 0.05 * std::cos(7 * x) + 0.02 * y;
        h_a[p]   = 0.9 + 0.04 * std::sin(5 * z);
        h_K[p]   = 0.1 * std::cos(4 * x + 3 * y);
        h_b0[p]  = 0.03 * y;
        h_b1[p]  = -0.03 * x;
        h_b2[p]  = 0.02 * z;
        h_v[p]   = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_chi, *d_a, *d_K, *d_b0, *d_b1, *d_b2, *d_v, *d_div;
  double *d_g0, *d_g1, *d_g2, *d_rhsp, *d_rhsm, *d_tl;
  cudaMalloc(&d_chi, N * sizeof(double));
  cudaMalloc(&d_a, N * sizeof(double));
  cudaMalloc(&d_K, N * sizeof(double));
  cudaMalloc(&d_b0, N * sizeof(double));
  cudaMalloc(&d_b1, N * sizeof(double));
  cudaMalloc(&d_b2, N * sizeof(double));
  cudaMalloc(&d_v, N * sizeof(double));
  cudaMalloc(&d_div, N * sizeof(double));
  cudaMalloc(&d_g0, N * sizeof(double));
  cudaMalloc(&d_g1, N * sizeof(double));
  cudaMalloc(&d_g2, N * sizeof(double));
  cudaMalloc(&d_rhsp, N * sizeof(double));
  cudaMalloc(&d_rhsm, N * sizeof(double));
  cudaMalloc(&d_tl, N * sizeof(double));

  cudaMemcpy(d_chi, h_chi.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);
  k_grad_dir<<<1, tb>>>(d_b0, 0, d_g0);
  k_grad_dir<<<1, tb>>>(d_b1, 1, d_g1);
  k_grad_dir<<<1, tb>>>(d_b2, 2, d_g2);
  k_sum3<<<1, tb>>>(d_g0, d_g1, d_g2, d_div);

  std::vector<double> rhsp(N), rhsm(N), tl(N);
  int all_pass = 1;

  // Gate A: δχ
  {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = h_chi[p] + eps * h_v[p];
      hm[p] = h_chi[p] - eps * h_v[p];
    }
    cudaMemcpy(d_chi, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_rhs<<<1, tb>>>(d_chi, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_rhsp);
    cudaMemcpy(d_chi, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_rhs<<<1, tb>>>(d_chi, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_rhsm);
    cudaMemcpy(d_chi, h_chi.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_tl_dchi<<<1, tb>>>(d_v, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_tl);
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd("dchi", rhsp, rhsm, tl, eps);
  }

  // Gate B: δα
  {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = h_a[p] + eps * h_v[p];
      hm[p] = h_a[p] - eps * h_v[p];
    }
    cudaMemcpy(d_a, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_rhs<<<1, tb>>>(d_chi, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_rhsp);
    cudaMemcpy(d_a, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_rhs<<<1, tb>>>(d_chi, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_rhsm);
    cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_tl_dalpha<<<1, tb>>>(d_chi, d_v, d_K, d_tl);
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd("dalpha", rhsp, rhsm, tl, eps);
  }

  // Gate C: δK
  {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = h_K[p] + eps * h_v[p];
      hm[p] = h_K[p] - eps * h_v[p];
    }
    cudaMemcpy(d_K, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_rhs<<<1, tb>>>(d_chi, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_rhsp);
    cudaMemcpy(d_K, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_rhs<<<1, tb>>>(d_chi, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_rhsm);
    cudaMemcpy(d_K, h_K.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_chi_tl_dK<<<1, tb>>>(d_chi, d_a, d_v, d_tl);
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd("dK", rhsp, rhsm, tl, eps);
  }

  std::printf("GATE %s\n", all_pass ? "PASS" : "FAIL");
  cudaFree(d_chi);
  cudaFree(d_a);
  cudaFree(d_K);
  cudaFree(d_b0);
  cudaFree(d_b1);
  cudaFree(d_b2);
  cudaFree(d_v);
  cudaFree(d_div);
  cudaFree(d_g0);
  cudaFree(d_g1);
  cudaFree(d_g2);
  cudaFree(d_rhsp);
  cudaFree(d_rhsm);
  cudaFree(d_tl);
  return all_pass ? 0 : 2;
}
