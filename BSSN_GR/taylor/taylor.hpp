// Truncated univariate Taylor ("jet") scalar for the B1 reference implementation.
// A Tay<K> holds the coefficients c[0..K] of f(eps) = sum_j c[j] eps^j (plain coefficients, NOT divided by factorials:
// the j-th derivative is j! * c[j]). Arithmetic is exact truncated power-series arithmetic, so evaluating the production
// pointwise algebra with this type yields the exact derivatives of the discrete RHS along a parameter direction.
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define TAY_HD __host__ __device__
#else
#define TAY_HD
#endif

template <int K>
struct Tay {
    double c[K + 1];
    TAY_HD Tay() { for (int i = 0; i <= K; i++) c[i] = 0.0; }
    TAY_HD Tay(double x) { c[0] = x; for (int i = 1; i <= K; i++) c[i] = 0.0; }
};

template <int K> TAY_HD inline Tay<K> operator-(const Tay<K>& a) { Tay<K> r; for (int i = 0; i <= K; i++) r.c[i] = -a.c[i]; return r; }
template <int K> TAY_HD inline Tay<K> operator+(const Tay<K>& a, const Tay<K>& b) { Tay<K> r; for (int i = 0; i <= K; i++) r.c[i] = a.c[i] + b.c[i]; return r; }
template <int K> TAY_HD inline Tay<K> operator-(const Tay<K>& a, const Tay<K>& b) { Tay<K> r; for (int i = 0; i <= K; i++) r.c[i] = a.c[i] - b.c[i]; return r; }
template <int K> TAY_HD inline Tay<K> operator*(const Tay<K>& a, const Tay<K>& b) {
    Tay<K> r;
    for (int i = 0; i <= K; i++) { double s = 0; for (int j = 0; j <= i; j++) s += a.c[j] * b.c[i - j]; r.c[i] = s; }
    return r;
}
template <int K> TAY_HD inline Tay<K> operator/(const Tay<K>& a, const Tay<K>& b) {
    Tay<K> r;
    for (int i = 0; i <= K; i++) { double s = a.c[i]; for (int j = 1; j <= i; j++) s -= b.c[j] * r.c[i - j]; r.c[i] = s / b.c[0]; }
    return r;
}
// scalar (double/int) mixes
template <int K> TAY_HD inline Tay<K> operator+(const Tay<K>& a, double b) { Tay<K> r = a; r.c[0] += b; return r; }
template <int K> TAY_HD inline Tay<K> operator+(double a, const Tay<K>& b) { return b + a; }
template <int K> TAY_HD inline Tay<K> operator-(const Tay<K>& a, double b) { Tay<K> r = a; r.c[0] -= b; return r; }
template <int K> TAY_HD inline Tay<K> operator-(double a, const Tay<K>& b) { Tay<K> r = -b; r.c[0] += a; return r; }
template <int K> TAY_HD inline Tay<K> operator*(const Tay<K>& a, double b) { Tay<K> r; for (int i = 0; i <= K; i++) r.c[i] = a.c[i] * b; return r; }
template <int K> TAY_HD inline Tay<K> operator*(double a, const Tay<K>& b) { return b * a; }
template <int K> TAY_HD inline Tay<K> operator/(const Tay<K>& a, double b) { Tay<K> r; for (int i = 0; i <= K; i++) r.c[i] = a.c[i] / b; return r; }
template <int K> TAY_HD inline Tay<K> operator/(double a, const Tay<K>& b) { return Tay<K>(a) / b; }
template <int K> TAY_HD inline Tay<K>& operator+=(Tay<K>& a, const Tay<K>& b) { for (int i = 0; i <= K; i++) a.c[i] += b.c[i]; return a; }

// y = x^e for real e (valid for x0 > 0, or integer e with x0 != 0): k x0 y_k = sum_{j=1..k} (e j - (k-j)) x_j y_{k-j}
template <int K> TAY_HD inline Tay<K> tpow(const Tay<K>& x, double e) {
    Tay<K> y;
    y.c[0] = std::pow(x.c[0], e);
    for (int k = 1; k <= K; k++) {
        double s = 0; for (int j = 1; j <= k; j++) s += (e * j - (k - j)) * x.c[j] * y.c[k - j];
        y.c[k] = s / (k * x.c[0]);
    }
    return y;
}
template <int K> TAY_HD inline Tay<K> pow(const Tay<K>& x, int e) { return tpow(x, (double)e); }
template <int K> TAY_HD inline Tay<K> pow(const Tay<K>& x, double e) { return tpow(x, e); }
template <int K> TAY_HD inline Tay<K> sqrt(const Tay<K>& x) { return tpow(x, 0.5); }
// y = exp(x): k y_k = sum_{j=1..k} j x_j y_{k-j}
template <int K> TAY_HD inline Tay<K> exp(const Tay<K>& x) {
    Tay<K> y; y.c[0] = std::exp(x.c[0]);
    for (int k = 1; k <= K; k++) { double s = 0; for (int j = 1; j <= k; j++) s += j * x.c[j] * y.c[k - j]; y.c[k] = s / k; }
    return y;
}
