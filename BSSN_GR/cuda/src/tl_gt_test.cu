/**
 * Tier 3.1 — gt00 TL scaffold vs FD (device PT).
 * Matches CodeGen/bssn_tl_gauge.py gt_rhs0 gates (weighted Lie + −2α At).
 *
 * Usage: tl_gt_test [eps]
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
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = f[t];
  __syncthreads();
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  double g, d2, ko;
  const double inv = 1.0 / DX;
  if (dir == 0) device::__pt_fused_x3<NX>(su, i, j, k, inv, &g, &d2, &ko);
  else if (dir == 1) device::__pt_fused_y3<NX>(su, i, j, k, inv, &g, &d2, &ko);
  else device::__pt_fused_z3<NX>(su, i, j, k, inv, &g, &d2, &ko);
  gout[p] = g;
}

__global__ void k_sum3(const double* a, const double* b, const double* c,
                       double* out) {
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  out[p] = a[p] + b[p] + c[p];
}

__global__ void k_gt_rhs(const double* gt0, const double* alpha,
                         const double* At0, const double* beta0,
                         const double* beta1, const double* beta2,
                         const double* divb, double* rhs) {
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
  rhs[p] = beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2 -
           TWO_THIRDS * gt0[p] * divb[p] - 2.0 * alpha[p] * At0[p] +
           KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_gt_tl_dgt(const double* v, const double* beta0,
                            const double* beta1, const double* beta2,
                            const double* divb, double* J) {
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
  J[p] = beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2 -
         TWO_THIRDS * v[p] * divb[p] + KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_gt_tl_dalpha(const double* At0, const double* va, double* J) {
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  J[p] = -2.0 * At0[p] * va[p];
}

__global__ void k_gt_tl_dAt(const double* alpha, const double* vAt, double* J) {
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  J[p] = -2.0 * alpha[p] * vAt[p];
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
      "tl_gt_test %s eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n", tag,
      eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;
  std::vector<double> h_gt(N), h_a(N), h_At(N), h_b0(N), h_b1(N), h_b2(N),
      h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_gt[p] = 1.0 + 0.04 * std::cos(6 * x) + 0.01 * y;
        h_a[p]  = 0.9 + 0.03 * std::sin(5 * z);
        h_At[p] = 0.02 * std::cos(4 * x + 2 * y);
        h_b0[p] = 0.03 * y;
        h_b1[p] = -0.03 * x;
        h_b2[p] = 0.02 * z;
        h_v[p]  = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_gt, *d_a, *d_At, *d_b0, *d_b1, *d_b2, *d_v, *d_div;
  double *d_g0, *d_g1, *d_g2, *d_rhsp, *d_rhsm, *d_tl;
  cudaMalloc(&d_gt, N * sizeof(double));
  cudaMalloc(&d_a, N * sizeof(double));
  cudaMalloc(&d_At, N * sizeof(double));
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

  cudaMemcpy(d_gt, h_gt.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_At, h_At.data(), N * sizeof(double), cudaMemcpyHostToDevice);
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

  auto fd_field = [&](std::vector<double>& h_field, double* d_field,
                      auto tl_launch, const char* tag) {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = h_field[p] + eps * h_v[p];
      hm[p] = h_field[p] - eps * h_v[p];
    }
    cudaMemcpy(d_field, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_gt_rhs<<<1, tb>>>(d_gt, d_a, d_At, d_b0, d_b1, d_b2, d_div, d_rhsp);
    cudaMemcpy(d_field, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_gt_rhs<<<1, tb>>>(d_gt, d_a, d_At, d_b0, d_b1, d_b2, d_div, d_rhsm);
    cudaMemcpy(d_field, h_field.data(), N * sizeof(double),
               cudaMemcpyHostToDevice);
    tl_launch();
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd(tag, rhsp, rhsm, tl, eps);
  };

  fd_field(h_gt, d_gt,
           [&] { k_gt_tl_dgt<<<1, tb>>>(d_v, d_b0, d_b1, d_b2, d_div, d_tl); },
           "dgt0");
  fd_field(h_a, d_a, [&] { k_gt_tl_dalpha<<<1, tb>>>(d_At, d_v, d_tl); },
           "dalpha");
  fd_field(h_At, d_At, [&] { k_gt_tl_dAt<<<1, tb>>>(d_a, d_v, d_tl); }, "dAt0");

  std::printf("GATE %s\n", all_pass ? "PASS" : "FAIL");
  for (auto* p : {d_gt, d_a, d_At, d_b0, d_b1, d_b2, d_v, d_div, d_g0, d_g1,
                  d_g2, d_rhsp, d_rhsm, d_tl})
    cudaFree(p);
  return all_pass ? 0 : 2;
}
