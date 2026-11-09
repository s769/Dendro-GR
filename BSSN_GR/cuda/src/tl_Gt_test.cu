/**
 * Tier 3.1 — Gt0 TL vs FD for the Lie+Gtk+KO slice of Gt_rhs
 * (Christoffel / β-seconds / αA terms held out — those need metric TL).
 *
 *   Gt_rhs0 ⊃ (β·∇)Gt0 + (2/3) Gt0 divβ − Gt_j ∂_j β0 + σ KO(Gt0)
 * with Gt1=Gt2=0 for the δGt0 smoke (Gtk≈Gt).
 *
 * δGt0: J = (β·∇)v + (2/3) v divβ − v ∂_0 β0 + σ KO(v)
 * δβ0:  J = v_β0 ∂_0 Gt0 − Gt0 ∂_0 v_β0   [divβ / other β held fixed in this dir]
 *
 * Usage: tl_Gt_test [eps]
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

// Lie+Gtk+KO slice; Gt1=Gt2=0 ⇒ only −Gt0 ∂0 β0 from −Gt_j ∂j β0.
__global__ void k_Gt_rhs(const double* Gt0, const double* beta0,
                         const double* beta1, const double* beta2,
                         const double* g0_b0, const double* divb, double* rhs) {
  __shared__ double su[N];
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  for (int t = tid; t < N; t += nthreads) su[t] = Gt0[t];
  __syncthreads();
  const int i = threadIdx.x + PW, j = threadIdx.y + PW, k = threadIdx.z + PW;
  if (i >= NX - PW || j >= NX - PW || k >= NX - PW) return;
  const int p = i + NX * (j + NX * k);
  const double inv = 1.0 / DX;
  double g0, g1, g2, dxx, dyy, dzz, k0, k1, k2;
  device::__pt_fused_x3<NX>(su, i, j, k, inv, &g0, &dxx, &k0);
  device::__pt_fused_y3<NX>(su, i, j, k, inv, &g1, &dyy, &k1);
  device::__pt_fused_z3<NX>(su, i, j, k, inv, &g2, &dzz, &k2);
  const double G = Gt0[p];
  rhs[p] = beta0[p] * g0 + beta1[p] * g1 + beta2[p] * g2 +
           TWO_THIRDS * G * divb[p] - G * g0_b0[p] +
           KO_SIGMA * (k0 + k1 + k2);
}

__global__ void k_Gt_tl_dGt(const double* v, const double* beta0,
                            const double* beta1, const double* beta2,
                            const double* g0_b0, const double* divb, double* J) {
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
         TWO_THIRDS * v[p] * divb[p] - v[p] * g0_b0[p] +
         KO_SIGMA * (k0 + k1 + k2);
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
      "tl_Gt_test %s eps=%.3e max|TL-FD|=%.3e rel=%.3e (fd=%.3e @p=%d)\n", tag,
      eps, max_abs, max_rel, max_fd, worst);
  const int pass = (max_rel < 1e-3) || (max_abs < 1e-8);
  std::printf("GATE_%s %s\n", tag, pass ? "PASS" : "FAIL");
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  const double eps = (argc > 1) ? std::atof(argv[1]) : 1e-6;
  std::vector<double> h_Gt(N), h_b0(N), h_b1(N), h_b2(N), h_v(N);
  for (int k = 0; k < NX; ++k)
    for (int j = 0; j < NX; ++j)
      for (int i = 0; i < NX; ++i) {
        const int p = i + NX * (j + NX * k);
        const double x = (i - NX / 2) * DX;
        const double y = (j - NX / 2) * DX;
        const double z = (k - NX / 2) * DX;
        h_Gt[p] = 0.04 * std::sin(4 * x + z);
        h_b0[p] = 0.03 * y;
        h_b1[p] = -0.03 * x;
        h_b2[p] = 0.02 * z;
        h_v[p]  = std::exp(-(x * x + y * y + z * z) / (4 * DX * DX));
      }

  double *d_Gt, *d_b0, *d_b1, *d_b2, *d_v, *d_div, *d_g0b0;
  double *d_t0, *d_t1, *d_t2, *d_g0Gt, *d_g0v, *d_rhsp, *d_rhsm, *d_tl;
  for (auto** pp :
       {&d_Gt, &d_b0, &d_b1, &d_b2, &d_v, &d_div, &d_g0b0, &d_t0, &d_t1, &d_t2,
        &d_g0Gt, &d_g0v, &d_rhsp, &d_rhsm, &d_tl})
    cudaMalloc(pp, N * sizeof(double));

  cudaMemcpy(d_Gt, h_Gt.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_b1.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_b2.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 tb(NX - 2 * PW, NX - 2 * PW, NX - 2 * PW);
  k_grad_dir<<<1, tb>>>(d_b0, 0, d_t0);
  k_grad_dir<<<1, tb>>>(d_b1, 1, d_t1);
  k_grad_dir<<<1, tb>>>(d_b2, 2, d_t2);
  k_sum3<<<1, tb>>>(d_t0, d_t1, d_t2, d_div);
  cudaMemcpy(d_g0b0, d_t0, N * sizeof(double), cudaMemcpyDeviceToDevice);

  std::vector<double> rhsp(N), rhsm(N), tl(N);
  int all_pass = 1;

  // δGt0
  {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = h_Gt[p] + eps * h_v[p];
      hm[p] = h_Gt[p] - eps * h_v[p];
    }
    cudaMemcpy(d_Gt, hp.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_Gt_rhs<<<1, tb>>>(d_Gt, d_b0, d_b1, d_b2, d_g0b0, d_div, d_rhsp);
    cudaMemcpy(d_Gt, hm.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_Gt_rhs<<<1, tb>>>(d_Gt, d_b0, d_b1, d_b2, d_g0b0, d_div, d_rhsm);
    cudaMemcpy(d_Gt, h_Gt.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_Gt_tl_dGt<<<1, tb>>>(d_v, d_b0, d_b1, d_b2, d_g0b0, d_div, d_tl);
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tl.data(), d_tl, N * sizeof(double), cudaMemcpyDeviceToHost);
    all_pass &= compare_tl_fd("dGt0", rhsp, rhsm, tl, eps);
  }

  // δβ0 — recompute ∂0 β0 and divβ after perturbing β0
  {
    std::vector<double> hp(N), hm(N);
    for (int p = 0; p < N; ++p) {
      hp[p] = h_b0[p] + eps * h_v[p];
      hm[p] = h_b0[p] - eps * h_v[p];
    }
    auto run_rhs = [&](const std::vector<double>& hb, double* out) {
      cudaMemcpy(d_b0, hb.data(), N * sizeof(double), cudaMemcpyHostToDevice);
      k_grad_dir<<<1, tb>>>(d_b0, 0, d_t0);
      k_grad_dir<<<1, tb>>>(d_b1, 1, d_t1);
      k_grad_dir<<<1, tb>>>(d_b2, 2, d_t2);
      k_sum3<<<1, tb>>>(d_t0, d_t1, d_t2, d_div);
      cudaMemcpy(d_g0b0, d_t0, N * sizeof(double), cudaMemcpyDeviceToDevice);
      k_Gt_rhs<<<1, tb>>>(d_Gt, d_b0, d_b1, d_b2, d_g0b0, d_div, out);
    };
    run_rhs(hp, d_rhsp);
    run_rhs(hm, d_rhsm);
    // restore β0 / div / ∂0β0 and build TL with frozen-divβ formula
    cudaMemcpy(d_b0, h_b0.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    k_grad_dir<<<1, tb>>>(d_Gt, 0, d_g0Gt);
    k_grad_dir<<<1, tb>>>(d_v, 0, d_g0v);
    cudaDeviceSynchronize();
    cudaMemcpy(rhsp.data(), d_rhsp, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsm.data(), d_rhsm, N * sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<double> gGt(N), gv(N), Gth(N);
    cudaMemcpy(gGt.data(), d_g0Gt, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gv.data(), d_g0v, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Gth.data(), d_Gt, N * sizeof(double), cudaMemcpyDeviceToHost);
    // J = v ∂0 Gt0 + (2/3) Gt0 ∂0 v − Gt0 ∂0 v = v ∂0 Gt0 − (1/3) Gt0 ∂0 v
    for (int p = 0; p < N; ++p)
      tl[p] = h_v[p] * gGt[p] - (1.0 / 3.0) * Gth[p] * gv[p];
    all_pass &= compare_tl_fd("dbeta0", rhsp, rhsm, tl, eps);
  }

  std::printf("GATE %s\n", all_pass ? "PASS" : "FAIL");
  for (auto* p : {d_Gt, d_b0, d_b1, d_b2, d_v, d_div, d_g0b0, d_t0, d_t1, d_t2,
                  d_g0Gt, d_g0v, d_rhsp, d_rhsm, d_tl})
    cudaFree(p);
  return all_pass ? 0 : 2;
}
