#!/usr/bin/env python3
"""Inspiral-window waveform comparison with data-analysis metrics (Gate A2).

usage: analyze_mismatch.py [--ridx 0] [--u0 60] [--u1 1e9] [--masses 60,100,200] RUN_COARSE ... RUN_FINEST
       (run dirs relative to ~/Dendro-GR-runs; ordered coarse -> fine, refinement factor 2 per level assumed)

What it does (psi4_22 at one extraction shell, retarded time u=t-r_ex; DUPLICATE SAMPLE TIMES ARE REMOVED):
  1. strain h22 by fixed-frequency integration (Reisswig-Pollney) with cutoff f0 = 0.8 * initial |f_gw| of the window, Tukey taper;
  2. mismatch 1 - max_{t0,phi0} <h_a|h_b>/sqrt(<h_a|h_a><h_b|h_b>) against the FINEST run for a flat PSD and an aLIGO-design
     PSD (analytic fit, Ajith 2011 ZDHP) at the requested total masses; single (2,2) mode, optimal orientation => proxy;
  3. residual dephasing/log-amplitude difference after aligning (time shift + phase) on the early window [u0, u0+50];
  4. with >=3 runs: observed convergence order of the aligned phase and log-amplitude (L2 norms, assuming factor-2 per level),
     the Richardson-extrapolated waveform and the mismatch of every run (and the error estimate of the finest) against it.
ASSUMPTIONS (flag in every report): network SNR 30 => thresholds 1/(2 rho^2)=5.6e-4 and D/(2 rho^2) with D=8 => 4.4e-3;
single (2,2) mode; strain from psi4 by FFI; aLIGO ZDHP analytic PSD; time in units of M, f_phys = f_M/(M_tot*4.925491e-6 s).
"""
import re, sys, argparse
import numpy as np

RAD = [50, 60, 70, 80, 90, 100]
MSUN_S = 4.925491e-6
R = "/home1/08435/srvenkat/Dendro-GR-runs/"


def load_psi4(run, ridx):
    t, z = [], []
    for l in open(R + run + "/dat/dgr_GW_l2_m2.dat").read().splitlines()[1:]:
        q = l.split()
        m = re.findall(r"\(([^,]+),([^)]+)\)", l)
        if len(q) < 3 or len(m) <= ridx:
            continue
        t.append(float(q[1])); z.append(complex(float(m[ridx][0]), float(m[ridx][1])))
    t = np.array(t); z = np.array(z)
    t, i = np.unique(t, return_index=True)          # de-duplicate by time (restarts), sorted
    return t - RAD[ridx], z[i]


def psd_aligo(f):
    """Analytic aLIGO zero-detuned high-power fit (Ajith 2011), valid above ~10 Hz; returns S_n(f) [1/Hz]."""
    x = np.maximum(f, 1.0) / 245.4
    s = 1e-48 * (0.0152 * x ** -4 + 0.2935 * x ** 2.25 + 2.7951 * x ** 1.5 - 6.5080 * x ** 0.75 + 17.7622)
    return np.where(f >= 10.0, s, 1e-40)


def resample(u, z, ug):
    A = np.interp(ug, u, np.abs(z)); ph = np.interp(ug, u, np.unwrap(np.angle(z)))
    return A * np.exp(1j * ph)


def ffi(psi, dt, f0):
    """h = double time integral of psi4 via fixed-frequency integration (two-sided FFT of a complex series)."""
    n = len(psi); f = np.fft.fftfreq(n, dt)
    ps = np.fft.fft(psi)
    den = (2 * np.pi * np.maximum(np.abs(f), f0)) ** 2
    return np.fft.ifft(-ps / den)


def tukey(n, alpha=0.2):
    w = np.ones(n); k = int(alpha * n / 2)
    if k > 0:
        r = 0.5 * (1 - np.cos(np.pi * np.arange(k) / k)); w[:k] = r; w[-k:] = r[::-1]
    return w


def match(ha, hb, dt, mass=None, f_lo_phys=10.0):
    """max over time shift and phase of the normalized overlap; mass=None => flat PSD (mass independent)."""
    n = len(ha); N = 4 * n                                   # zero-pad to resolve time shifts
    f = np.fft.fftfreq(N, dt)
    A = np.fft.fft(ha, N); B = np.fft.fft(hb, N)
    if mass is None:
        S = np.ones(N); band = np.ones(N, bool)
    else:
        fp = np.abs(f) / (mass * MSUN_S); S = psd_aligo(fp); band = fp >= f_lo_phys
    W = np.where(band, 1.0 / S, 0.0)
    inner = lambda X, Y: np.sum(np.conj(X) * Y * W).real
    nA, nB = np.sqrt(inner(A, A)), np.sqrt(inner(B, B))
    if nA == 0 or nB == 0:
        return np.nan
    ov = np.abs(np.fft.ifft(np.conj(A) * B * W)) * N
    return float(ov.max() / (nA * nB))


def align(href, h, ug, win):
    """time shift (parabolic peak of the cross-correlation) + phase so that h best matches href; returns aligned complex h."""
    dt = ug[1] - ug[0]; N = 4 * len(ug)
    A = np.fft.fft(href, N); B = np.fft.fft(h, N)
    c = np.fft.ifft(np.conj(A) * B) * N; k = int(np.argmax(np.abs(c)))
    if k > N // 2: k -= N
    sh = -k * dt                                               # advance h by sh (h(u+sh)... applied via FFT phase ramp)
    f = np.fft.fftfreq(N, dt)
    hs = np.fft.ifft(np.fft.fft(h, N) * np.exp(2j * np.pi * f * (-sh)))[:len(ug)]
    ph = np.angle(np.sum(np.conj(href) * hs)); return hs * np.exp(-1j * ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--ridx", type=int, default=0)
    ap.add_argument("--u0", type=float, default=60.0); ap.add_argument("--u1", type=float, default=1e9)
    ap.add_argument("--masses", default="60,100,200"); ap.add_argument("--dt", type=float, default=0.5)
    a = ap.parse_args()
    masses = [float(x) for x in a.masses.split(",")]
    data = [load_psi4(r, a.ridx) for r in a.runs]
    u1 = min(a.u1, *[d[0][-1] for d in data]); u0 = a.u0
    n = int((u1 - u0) / a.dt); ug = u0 + a.dt * np.arange(n)
    psi = [resample(u, z, ug) for u, z in data]
    # initial GW frequency of the finest run for the FFI cutoff
    ph = np.unwrap(np.angle(psi[-1])); f_init = abs((ph[10] - ph[0]) / (10 * a.dt)) / (2 * np.pi); f0 = 0.8 * f_init
    w = tukey(n)
    h = [ffi(p, a.dt, f0) * w for p in psi]
    print(f"window u in [{u0:.0f},{u1:.0f}] r_ex={RAD[a.ridx]} dt={a.dt} n={n}; f_gw(init)={f_init:.4f}/M f0={f0:.4f}/M; "
          f"SNR-30 thresholds: 1/(2 rho^2)=5.6e-4, D=8: 4.4e-3  [ASSUMPTIONS: single 22 mode, FFI strain, aLIGO ZDHP fit]")
    ref = h[-1]
    print("\n(2) mismatch vs finest run (%s):" % a.runs[-1])
    for i, r in enumerate(a.runs[:-1]):
        row = [f"flat {1 - match(ref, h[i], a.dt):.3e}"]
        for m in masses:
            row.append(f"aLIGO M={m:.0f}: {1 - match(ref, h[i], a.dt, m):.3e}")
        print(f"  {r}: " + " | ".join(row))
    # (3) residual dephasing / log-amp after aligning on the early window
    print("\n(3) residual phase / log-amplitude difference vs finest after (t,phi) alignment on the first 50 M of the window:")
    ew = int(50 / a.dt)
    def early_align(hr, hh):
        # time shift + phase from the early window only, applied to the whole series
        N = 4 * ew; f = np.fft.fftfreq(N, a.dt)
        A = np.fft.fft(hr[:ew], N); B = np.fft.fft(hh[:ew], N)
        c = np.fft.ifft(np.conj(A) * B) * N; k = int(np.argmax(np.abs(c)))
        if k > N // 2: k -= N
        Nf = len(hh); ff = np.fft.fftfreq(2 * Nf, a.dt)
        hs = np.fft.ifft(np.fft.fft(hh, 2 * Nf) * np.exp(2j * np.pi * ff * (k * a.dt)))[:Nf]
        pp = np.angle(np.sum(np.conj(hr[:ew]) * hs[:ew])); return hs * np.exp(-1j * pp)
    for i, r in enumerate(a.runs[:-1]):
        hs = early_align(ref, h[i])
        dph = np.unwrap(np.angle(hs)) - np.unwrap(np.angle(ref)); dph -= dph[ew // 2]
        dla = np.log(np.abs(hs) + 1e-300) - np.log(np.abs(ref) + 1e-300)
        pts = [ug[int(x)] for x in (n // 4, n // 2, 3 * n // 4, n - 1)]
        print(f"  {r}: " + "  ".join(f"u={ug[int(x)]:.0f}: dphi={dph[int(x)]:+.3f} dlnA={dla[int(x)]:+.3f}" for x in (n // 4, n // 2, 3 * n // 4, n - 1)))
    # (4) convergence order + Richardson
    if len(a.runs) >= 3:
        print("\n(4) convergence (aligned to the finest; amplitude/phase L2 norms over the window, factor-2 refinement assumed):")
        hs = [early_align(ref, h[i]) for i in range(len(h))]
        ph = [np.unwrap(np.angle(x)) for x in hs]; la = [np.log(np.abs(x) + 1e-300) for x in hs]
        for i in range(len(h) - 2):
            d1p = np.linalg.norm(ph[i] - ph[i + 1]); d2p = np.linalg.norm(ph[i + 1] - ph[i + 2])
            d1a = np.linalg.norm(la[i] - la[i + 1]); d2a = np.linalg.norm(la[i + 1] - la[i + 2])
            op = np.log2(d1p / d2p) if d2p > 0 else np.nan; oa = np.log2(d1a / d2a) if d2a > 0 else np.nan
            print(f"  levels ({a.runs[i]},{a.runs[i+1]},{a.runs[i+2]}): |dphi| {d1p:.3g} -> {d2p:.3g} (order {op:.2f}); |dlnA| {d1a:.3g} -> {d2a:.3g} (order {oa:.2f})")
        p = max(1.0, min(8.0, op)) if np.isfinite(op) else 6.0
        phinf = ph[-1] + (ph[-1] - ph[-2]) / (2 ** p - 1); lainf = la[-1] + (la[-1] - la[-2]) / (2 ** p - 1)
        hinf = np.exp(lainf) * np.exp(1j * phinf) * np.where(w > 0, 1, 0)
        print(f"  Richardson with order p={p:.2f} ('error estimate' = mismatch of each run vs h_inf):")
        for i, r in enumerate(a.runs):
            row = [f"flat {1 - match(hinf, hs[i], a.dt):.3e}"] + [f"M={m:.0f}: {1 - match(hinf, hs[i], a.dt, m):.3e}" for m in masses]
            print(f"    {r}: " + " | ".join(row))


if __name__ == "__main__":
    main()
