#!/usr/bin/env python3
"""FD derivative of psi4_22 phase/amplitude wrt eps. usage: analyze_wave_fd.py OUTDIR eps1 eps2 ..."""
import sys, re, numpy as np
from pathlib import Path
O = Path(sys.argv[1]); epss = [float(e) for e in sys.argv[2:]]
def load(tag, lm="l2_m2"):
    t, z = [], []
    for l in (O/tag/"dat"/f"bssn_profile_GW_{lm}.dat").read_text().splitlines()[1:]:
        p = l.split()
        if len(p) < 3: continue
        t.append(float(p[1]))
        z.append([complex(*map(float, re.findall(r"[-+0-9.eE]+", c))) for c in p[2:]])
    return np.array(t), np.array(z)   # z[time, radius]
t0, zf = load("fid")
nr = zf.shape[1]
def phase(z): return np.unwrap(np.angle(z), axis=0)
print("t range", t0[0], t0[-1], "n", len(t0))
for ir in range(nr):
    print(f"--- radius idx {ir}: |psi4_22| max {abs(zf[:,ir]).max():.3e}, total phase change {np.ptp(phase(zf)[:,ir]):.3f} rad")
ir = int(Path(sys.argv[0]).stem == "" ) # placeholder
for ir in [0, nr//2, nr-1]:
    print(f"=== radius idx {ir}: central FD vs eps (rows: window-averaged over last 25% of run)")
    n0 = int(0.75*len(t0))
    res = {}
    for e in epss:
        tp, zp = load(f"p_{e}"); tm, zm = load(f"m_{e}")
        n = min(len(tp), len(tm), len(t0))
        dph = (phase(zp[:n])[:, ir] - phase(zm[:n])[:, ir]) / (2*e)
        dA  = (abs(zp[:n, ir]) - abs(zm[:n, ir])) / (2*e)
        # also complex derivative normalized by fiducial amplitude
        dz  = (zp[:n, ir] - zm[:n, ir]) / (2*e)
        res[e] = (dph, dA, dz)
        k = slice(n0, n)
        print(f" eps={e:g}: dphi/dm mean(last25%)={dph[k].mean():.5e}  end={dph[-1]:.5e} | dA/dm end={dA[-1]:.4e} | |dz|/|z| end={abs(dz[-1])/abs(zf[n-1,ir]):.4e}")
    if len(epss) >= 2:
        a, b = epss[0], epss[1]; n = min(len(res[a][0]), len(res[b][0]))
        d = res[a][2][:n] - res[b][2][:n]
        print(f" eps-plateau: ||dz(eps{a:g})-dz(eps{b:g})|| / ||dz(eps{a:g})|| = {np.linalg.norm(d)/np.linalg.norm(res[a][2][:n]):.3e}")
