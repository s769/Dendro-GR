/**
 * Tier 3.1 — At00 TL vs FD for the algebraic+Lie slice of At_rhs
 * (Ricci / DiDj(α) / TF held out — those need metric TL).
 *
 *   At_rhs00 ⊃ (β·∇)At00 + w At00 divβ + α (K At00 − 2 A0k A^k_0) + σ KO
 * with A0k A^k_0 ≈ At0² for diagonal smoke (igt≈δ; At off-diag=0).
 *
 * δAt0: J = (β·∇)v + w v divβ + α (K v − 4 At0 v) + σ KO(v)   [igt=1]
 * δα:   J = v_α (K At0 − 2 At0²)
 *
 * Usage: tl_At_test [eps]
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

__global__ void k_At_rhs(const double* At0, const double* alpha, const double* K,
                         const double* beta0, const double* beta1,
                         const double* beta2, const double* divb, double* rhs) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = At0[t];
  __syncthreads();
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  const double A = At0[p];
  rhs[p] = beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2 -
           TWO_THIRDS * A * divb[p] + alpha[p] * (K[p] * A - 2.0 * A * A) +
           KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_At_tl_dAt(const double* v, const double* At0,
                            const double* alpha, const double* K,
                            const double* beta0, const double* beta1,
                            const double* beta2, const double* divb, double* J) {
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
         TWO_THIRDS * v[p] * divb[p] +
         alpha[p] * (K[p] * v[p] - 4.0 * At0[p] * v[p]) +
         KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_At_tl_dalpha(const double* At0, const double* K,
                               const double* va, double* J) {
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  const double A = At0[p];
  J[p] = va[p] * (K[p] * A - 2.0 * A * A);
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
      "tl_At_test %s eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n", tag,
      eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;
  std::vector<double> h_At(N), h_a(N), h_K(N), h_b0(N), h_b1(N), h_b2(N),
      h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_At[p] = 0.05 * std::cos(5 * x + 2 * z);
        h_a[p]  = 0.9 + 0.03 * std::sin(4 * y);
        h_K[p]  = 0.08 * std::cos(3 * x);
        h_b0[p] = 0.03 * y;
        h_b1[p] = -0.03 * x;
        h_b2[p] = 0.02 * z;
        h_v[p]  = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_At, *d_a, *d_K, *d_b0, *d_b1, *d_b2, *d_v, *d_div;
  double *d_t0, *d_t1, *d_t2, *d_rhsp, *d_rhsm, *d_tl;
  for (auto** pp : {&d_At, &d_a, &d_K, &d_b0, &d_b1, &d_b2, &d_v, &d_div, &d_t0,
                    &d_t1, &d_t2, &d_rhsp, &d_rhsm, &d_tl})
    cudaMalloc(pp, N * sizeof(double));

  cudaMemcpy(d_At, h_At.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);
  k_grad_dir<<<1, tb>>>(d_b0, 0, d_t0);
  k_grad_dir<<<1, tb>>>(d_b1, 1, d_t1);
  k_grad_dir<<<1, tb>>>(d_b2, 2, d_t2);
  k_sum3<<<1, tb>>>(d_t0, d_t1, d_t2, d_div);

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
    k_At_rhs<<<1, tb>>>(d_At, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_rhsp);
    cudaMemcpy(df, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_At_rhs<<<1, tb>>>(d_At, d_a, d_K, d_b0, d_b1, d_b2, d_div, d_rhsm);
    cudaMemcpy(df, hf.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    tl_launch();
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd(tag, rhsp, rhsm, tl, eps);
  };

  fd(h_At, d_At,
     [&] {
       k_At_tl_dAt<<<1, tb>>>(d_v, d_At, d_a, d_K, d_b0, d_b1, d_b2, d_div,
                              d_tl);
     },
     "dAt0");
  fd(h_a, d_a, [&] { k_At_tl_dalpha<<<1, tb>>>(d_At, d_K, d_v, d_tl); },
     "dalpha");

  std::printf("GATE %s\n", all_pass ? "PASS" : "FAIL");
  for (auto* p : {d_At, d_a, d_K, d_b0, d_b1, d_b2, d_v, d_div, d_t0, d_t1, d_t2,
                  d_rhsp, d_rhsm, d_tl})
    cudaFree(p);
  return all_pass ? 0 : 2;
}
