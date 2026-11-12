// GPU microbenchmark of the Taylor-typed pointwise BSSN RHS algebra (generated body, per-point jet of order K).
// usage: jet_gpu_bench [npts=4194304] [reps=20]   prints ns/point and ns/point/K0 for K=0..3; register/spill via -Xptxas -v
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "taylor.hpp"
#include "taylor_names.inc"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// structure-of-arrays: for each input array name, K+1 coefficient planes of npts doubles
template <int K>
__global__ void __launch_bounds__(128) jet_rhs(const double* __restrict__ in, double* __restrict__ out, long npts) {
    typedef Tay<K> T;
    long pp = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (pp >= npts) return;
    const double dx_i = 0.01, dt = 0.01, h_ssl = 0.6, sig_ssl = 1.0, t = 0.7, BSSN_CAHD_C = 0.02, eta = 1.3;
    const unsigned lambda[4] = {1, 1, 1, 1};
    const double lambda_f[2] = {1.0, 0.0};
    int a = 0;
#define LDF(n, v) T n##_T; for (int q = 0; q <= K; q++) n##_T.c[q] = in[((long)(a * (K + 1) + q)) * npts + pp]; a++;
    TAYLOR_FIELD_NAMES(LDF)
#undef LDF
#define LDD(n) T n##_T; for (int q = 0; q <= K; q++) n##_T.c[q] = in[((long)(a * (K + 1) + q)) * npts + pp]; a++;
    TAYLOR_DERIV_NAMES(LDD)
#undef LDD
#define OD(n, v, f) T OUT_##n;
    TAYLOR_OUT_NAMES(OD)
#undef OD
#include "taylor_point.inc"
    int o = 0;
#define WR(n, v, f) for (int q = 0; q <= K; q++) out[((long)(o * (K + 1) + q)) * npts + pp] = OUT_##n.c[q]; o++;
    TAYLOR_OUT_NAMES(WR)
#undef WR
}

#define COUNT(...) +1
template <int K> double run(long npts, int reps) {
    int nin = 0;
#define C1(n, v) nin++;
    TAYLOR_FIELD_NAMES(C1)
#undef C1
#define C2(n) nin++;
    TAYLOR_DERIV_NAMES(C2)
#undef C2
    int nout = 0;
#define C3(n, v, f) nout++;
    TAYLOR_OUT_NAMES(C3)
#undef C3
    size_t bi = (size_t)nin * (K + 1) * npts, bo = (size_t)nout * (K + 1) * npts;
    std::vector<double> h(bi);
    srand(1);
    for (size_t i = 0; i < bi; i++) h[i] = 0.01 * (rand() / (double)RAND_MAX - 0.5);
    // plausible base state: alpha, chi, gt diag, K=0 near 1: set coefficient-0 planes of a few fields
    double *din, *dout; CK(cudaMalloc(&din, bi * 8)); CK(cudaMalloc(&dout, bo * 8));
    CK(cudaMemcpy(din, h.data(), bi * 8, cudaMemcpyHostToDevice));
    // field order is the macro order; set the coefficient-0 plane of alpha/chi/gt0/gt3/gt5 to 1
    int a = 0;
#define FS(n, v) { std::vector<double> one(npts, 1.0); if (!strcmp(#n,"alpha")||!strcmp(#n,"chi")||!strcmp(#n,"gt0")||!strcmp(#n,"gt3")||!strcmp(#n,"gt5")) CK(cudaMemcpy(din + (size_t)(a*(K+1))*npts, one.data(), npts*8, cudaMemcpyHostToDevice)); a++; }
    TAYLOR_FIELD_NAMES(FS)
#undef FS
    int bs = 128; long nb = (npts + bs - 1) / bs;
    jet_rhs<K><<<nb, bs>>>(din, dout, npts); CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; r++) jet_rhs<K><<<nb, bs>>>(din, dout, npts);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, jet_rhs<K>);
    double nspp = ms * 1e6 / reps / npts;
    printf("K=%d: %.3f ns/point (%.2f Gpts/s)  regs=%d local(spill/stack)=%zu B  bytes/pt moved=%d\n", K, nspp, 1.0 / nspp, fa.numRegs, (size_t)fa.localSizeBytes, (int)(8 * (nin + nout) * (K + 1)));
    cudaFree(din); cudaFree(dout);
    return nspp;
}
int main(int argc, char** argv) {
    long npts = argc > 1 ? atol(argv[1]) : 4194304; int reps = argc > 2 ? atoi(argv[2]) : 20;
    printf("npts=%ld reps=%d\n", npts, reps);
    double a0 = run<0>(npts, reps), a1 = run<1>(npts, reps), a2 = run<2>(npts, reps), a3 = run<3>(npts, reps);
    printf("ratios vs K=0: K=1 %.2f  K=2 %.2f  K=3 %.2f\n", a1 / a0, a2 / a0, a3 / a0);
    return 0;
}
