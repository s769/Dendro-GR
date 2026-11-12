#!/usr/bin/env python3
"""A3: finite-extraction-radius systematics. For ONE run, mismatch of the (2,2) strain at shell r_i against the r=100 shell and
against a 1/r-extrapolated waveform, on retarded time u=t-r, aligned in (t,phi) (analyze_mismatch machinery).
usage: analyze_extraction_radius.py RUN [--u0 60] [--u1 125]"""
import sys, argparse, numpy as np
sys.path.insert(0, '.')
import analyze_mismatch as am
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--u0', type=float, default=60); ap.add_argument('--u1', type=float, default=125); ap.add_argument('--dt', type=float, default=0.5)
a = ap.parse_args()
data = [am.load_psi4(a.run, i) for i in range(6)]
u1 = min(a.u1, *[d[0][-1] for d in data]); n = int((u1 - a.u0) / a.dt); ug = a.u0 + a.dt * np.arange(n)
psi = [am.resample(u, z, ug) for u, z in data]
ph = np.unwrap(np.angle(psi[0])); f0 = 0.8 * abs((ph[10] - ph[0]) / (10 * a.dt)) / (2 * np.pi)
w = am.tukey(n); h = [am.ffi(p, a.dt, f0) * w for p in psi]
R = np.array(am.RAD, float)
print(f"run {a.run}: window u in [{a.u0:.0f},{u1:.0f}] ({n} samples); amplitude scale check |psi4| at u=80: " + ", ".join(f"r={int(r)}: {abs(p[int((80-a.u0)/a.dt)]):.3e}" for r, p in zip(R, psi)))
# r*psi4 convention: test whether |psi4| ~ 1/r
print("   |psi4|*r at u=80: " + ", ".join(f"{abs(p[int((80-a.u0)/a.dt)])*r:.4e}" for r, p in zip(R, psi)))
ref = h[5]
print("mismatch of shell r_i vs r=100 shell (flat | aLIGO M=60,100,200):")
for i in range(5):
    print(f"  r={int(R[i])}: flat {1-am.match(ref,h[i],a.dt):.3e} | " + " | ".join(f"{1-am.match(ref,h[i],a.dt,m):.3e}" for m in (60, 100, 200)))
# 1/r polynomial extrapolation of aligned log(r|h|) and phase (degree 1 and 2 in 1/r, fit over r=60..100; 2pi branch fixed vs r=100)
hs = [am.align(ref, h[i], ug, None) for i in range(6)]
ph_ = np.array([np.unwrap(np.angle(x)) for x in hs])
for i in range(6): ph_[i] -= 2 * np.pi * np.round((ph_[i][n // 2] - ph_[5][n // 2]) / (2 * np.pi))
la = np.array([np.log(np.abs(x) * R[i] + 1e-300) for i, x in enumerate(hs)])
x = 1.0 / R[1:]
def extrap(Y, deg):
    return np.array([np.polyfit(x, Y[1:, j], deg)[-1] for j in range(Y.shape[1])])
hinf = {d: np.exp(extrap(la, d)) * np.exp(1j * extrap(ph_, d)) * (w > 0) for d in (1, 2)}
for d in (1, 2):
    print(f"mismatch of each shell vs the degree-{d} 1/r-extrapolated (r->inf) waveform (flat | aLIGO M=60,100,200):")
    for i in range(6):
        print(f"  r={int(R[i])}: flat {1-am.match(hinf[d],hs[i],a.dt):.3e} | " + " | ".join(f"{1-am.match(hinf[d],hs[i],a.dt,m):.3e}" for m in (60, 100, 200)))
print("extrapolation uncertainty: mismatch(deg1 vs deg2 extrapolant): flat %.3e | " % (1 - am.match(hinf[2], hinf[1], a.dt)) + " | ".join(f"{1-am.match(hinf[2],hinf[1],a.dt,m):.3e}" for m in (60, 100, 200)))
for d in (1, 2):
    print(f"  deg{d}: dphi(r=100 shell - extrapolant) at u=" + ", ".join(f"{ug[k]:.0f}: {ph_[5][k]-np.unwrap(np.angle(hinf[d]))[k]:+.4f}" for k in (n//4, n//2, 3*n//4, n-1)) + "; dlnA at same: " + ", ".join(f"{np.log(abs(hs[5][k])*R[5]/abs(hinf[d][k])):+.4f}" for k in (n//4, n//2, 3*n//4, n-1)))
