/**
 * Tier 3.1 #10 — Γ-driver B⁰ TL vs FD (device PT).
 *
 * Production interior (λ=[1,1,1,1], const η):
 *   B_rhs0 = -η B0 - λ3 (β·∇)Gt0 + Gt_rhs0 + λ2 (β·∇)B0 + σ_KO KO(B0)
 *
 * Gate A — direction v=δB0 (β, Gt fixed; Gt_rhs0 / (β·∇)Gt cancel in FD):
 *   JB = -η v + λ2 (β·∇)v + σ_KO KO(v)
 *
 * Gate B — direction w=δβ0 (B0, β1, β2, Gt fixed):
 *   JB = λ2 w ∂0 B0 - λ3 w ∂0 Gt0
 *
 * Usage: tl_Bdriver_test [eps]
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
constexpr double LAMBDA2 = 1.0;
constexpr double LAMBDA3 = 1.0;
constexpr double ETA = 2.0;
constexpr double KO_SIGMA = 0.4;

__global__ void k_Brhs0(const double* B0, const double* beta0,
                        const double* beta1, const double* beta2,
                        const double* Gt0, const double* Gt_rhs0,
                        double* B_rhs0) {
  __shared__ double sB[N];
  __shared__ double sGt[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) {
    sB[t]  = B0[t];
    sGt[t] = Gt0[t];
  }
  __syncthreads();

  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;

  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double gB0, gB1, gB2, dxx, dyy, dzz, k0, k1, k2;
  double gGt0, gGt1, gGt2, d2x, d2y, d2z, kGt0, kGt1, kGt2;
  device::__pt_fused_x3<NX>(sB, i, j, k, inv, &gB0, &dxx, &k0);
  device::__pt_fused_y3<NX>(sB, i, j, k, inv, &gB1, &dyy, &k1);
  device::__pt_fused_z3<NX>(sB, i, j, k, inv, &gB2, &dzz, &k2);
  device::__pt_fused_x3<NX>(sGt, i, j, k, inv, &gGt0, &d2x, &kGt0);
  device::__pt_fused_y3<NX>(sGt, i, j, k, inv, &gGt1, &d2y, &kGt1);
  device::__pt_fused_z3<NX>(sGt, i, j, k, inv, &gGt2, &d2z, &kGt2);
  (void)kGt0;
  (void)kGt1;
  (void)kGt2;
  B_rhs0[p] = -ETA * B0[p] - LAMBDA3 * (beta0[p] * gGt0 + beta1[p] * gGt1 +
                                        beta2[p] * gGt2) +
              Gt_rhs0[p] +
              LAMBDA2 * (beta0[p] * gB0 + beta1[p] * gB1 + beta2[p] * gB2) +
              KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_B0_tl(const double* v, const double* beta0,
                        const double* beta1, const double* beta2, double* JB) {
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
  JB[p] = -ETA * v[p] +
          LAMBDA2 * (beta0[p] * gv0 + beta1[p] * gv1 + beta2[p] * gv2) +
          KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_grad0(const double* f, double* g0) {
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
  double gx, dxx, kx;
  device::__pt_fused_x3<NX>(su, i, j, k, 1.0 / DX, &gx, &dxx, &kx);
  g0[p] = gx;
}

__global__ void k_B0_tl_dbeta0(const double* w, const double* g0_B0,
                              const double* g0_Gt0, double* JB) {
  const int i = threadIdx.x + PW;
  const int j = threadIdx.y + PW;
  const int k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  JB[p] = LAMBDA2 * w[p] * g0_B0[p] - LAMBDA3 * w[p] * g0_Gt0[p];
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
      "tl_Bdriver_test %s eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n",
      tag, eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;

  std::vector<double> h_B0(N), h_b0(N), h_b1(N), h_b2(N), h_Gt0(N), h_Gtr(N),
      h_v(N), h_w(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_B0[p]  = 0.05 * std::cos(8 * x + 4 * z);
        h_b0[p]  = 0.03 * y + 0.01 * std::sin(12 * x);
        h_b1[p]  = -0.03 * x;
        h_b2[p]  = 0.02 * z;
        h_Gt0[p] = 0.04 * std::sin(6 * y) + 0.01 * x;
        h_Gtr[p] = 0.02 * std::cos(5 * z);  // fake Gt_rhs0 background
        h_v[p]   = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
        h_w[p]   = 0.7 * h_v[p] + 0.1 * std::sin(9 * x);
      }

  double *d_B0, *d_b0, *d_b1, *d_b2, *d_Gt0, *d_Gtr, *d_v, *d_w;
  double *d_rhs0, *d_rhsp, *d_rhsm, *d_tl, *d_g0B, *d_g0Gt;
  cudaMalloc(&d_B0, N * sizeof(double));
  cudaMalloc(&d_b0, N * sizeof(double));
  cudaMalloc(&d_b1, N * sizeof(double));
  cudaMalloc(&d_b2, N * sizeof(double));
  cudaMalloc(&d_Gt0, N * sizeof(double));
  cudaMalloc(&d_Gtr, N * sizeof(double));
  cudaMalloc(&d_v, N * sizeof(double));
  cudaMalloc(&d_w, N * sizeof(double));
  cudaMalloc(&d_rhs0, N * sizeof(double));
  cudaMalloc(&d_rhsp, N * sizeof(double));
  cudaMalloc(&d_rhsm, N * sizeof(double));
  cudaMalloc(&d_tl, N * sizeof(double));
  cudaMalloc(&d_g0B, N * sizeof(double));
  cudaMalloc(&d_g0Gt, N * sizeof(double));

  cudaMemcpy(d_B0, h_B0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Gt0, h_Gt0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Gtr, h_Gtr.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, h_w.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);

  // ---- Gate A: δB0 ----
  std::vector<double> h_Bp(N), h_Bm(N);
  for (int p = 0; p < N; ++p) {
    h_Bp[p] = h_B0[p] + eps * h_v[p];
    h_Bm[p] = h_B0[p] - eps * h_v[p];
  }
  cudaMemcpy(d_B0, h_Bp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_Brhs0<<<1, tb>>>(d_B0, d_b0, d_b1, d_b2, d_Gt0, d_Gtr, d_rhsp);
  cudaMemcpy(d_B0, h_Bm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_Brhs0<<<1, tb>>>(d_B0, d_b0, d_b1, d_b2, d_Gt0, d_Gtr, d_rhsm);
  cudaMemcpy(d_B0, h_B0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_B0_tl<<<1, tb>>>(d_v, d_b0, d_b1, d_b2, d_tl);
  cudaDeviceSynchronize();

  std::vector<double> rhsp(N), rhsm(N), tl(N);
  cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
  const int passA = compare_tl_fd("dB0", rhsp, rhsm, tl, eps);

  // ---- Gate B: δβ0 ----
  k_grad0<<<1, tb>>>(d_B0, d_g0B);
  k_grad0<<<1, tb>>>(d_Gt0, d_g0Gt);
  std::vector<double> h_bp(N), h_bm(N);
  for (int p = 0; p < N; ++p) {
    h_bp[p] = h_b0[p] + eps * h_w[p];
    h_bm[p] = h_b0[p] - eps * h_w[p];
  }
  cudaMemcpy(d_b0, h_bp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_Brhs0<<<1, tb>>>(d_B0, d_b0, d_b1, d_b2, d_Gt0, d_Gtr, d_rhsp);
  cudaMemcpy(d_b0, h_bm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_Brhs0<<<1, tb>>>(d_B0, d_b0, d_b1, d_b2, d_Gt0, d_Gtr, d_rhsm);
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  k_B0_tl_dbeta0<<<1, tb>>>(d_w, d_g0B, d_g0Gt, d_tl);
  cudaDeviceSynchronize();

  cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
  const int passB = compare_tl_fd("dbeta0", rhsp, rhsm, tl, eps);

  std::printf("GATE %s\n", (passA && passB) ? "PASS" : "FAIL");

  cudaFree(d_B0);
  cudaFree(d_b0);
  cudaFree(d_b1);
  cudaFree(d_b2);
  cudaFree(d_Gt0);
  cudaFree(d_Gtr);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_rhs0);
  cudaFree(d_rhsp);
  cudaFree(d_rhsm);
  cudaFree(d_tl);
  cudaFree(d_g0B);
  cudaFree(d_g0Gt);
  return (passA && passB) ? 0 : 2;
}
