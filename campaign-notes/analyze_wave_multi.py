#!/usr/bin/env python3
"""Combine waveform-FD run dirs. usage: analyze_wave_multi.py LABEL=DIR [LABEL=DIR ...] [--tmin T] [--ridx I]
Each DIR holds fid/ and p_<eps>/ m_<eps>/. Prints, per label and radius, dz/dm(eps) = (z_p - z_m)/2eps (psi4_22 complex),
eps-plateau (rel L2 of dz(eps)-dz(eps_ref)) on window t>=tmin, and cross-label dz agreement (interp to first label's times)."""
import sys, re, numpy as np
from pathlib import Path
args=[a for a in sys.argv[1:] if '=' in a and not a.startswith('--')]
opt=dict(zip(sys.argv[1:], sys.argv[2:]))
tmin=float(opt.get('--tmin',20)); ridxs=[int(x) for x in opt.get('--ridx','0,2,5').split(',')]
def load(d,tag):
    t,z=[],[]
    for l in (Path(d)/tag/'dat'/'bssn_profile_GW_l2_m2.dat').read_text().splitlines()[1:]:
        p=l.split()
        if len(p)<3: continue
        t.append(float(p[1])); z.append([complex(*map(float,re.findall(r'[-+0-9.eE]+',c))) for c in p[2:]])
    return np.array(t),np.array(z)
D={}
for a in args:
    lab,d=a.split('=',1); D[lab]={}
    for p in sorted(Path(d).glob('p_*')):
        e=p.name[2:]
        if (Path(d)/f'm_{e}'/'dat').exists():
            t,zp=load(d,p.name); tm,zm=load(d,f'm_{e}'); n=min(len(t),len(tm))
            D[lab][float(e)]=(t[:n],(zp[:n]-zm[:n])/(2*float(e)))
def rel(a,b): return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)
for lab,eps in D.items():
    es=sorted(eps)
    for ir in ridxs:
        print(f'[{lab}] radius idx {ir}, window t>={tmin}')
        ref=es[len(es)//2]
        for e in es:
            t,dz=eps[e]; tr,dzr=eps[ref]; n=min(len(t),len(tr)); k=(t[:n]>=tmin)
            print(f'   eps={e:<8g} |dz|window-rms={np.sqrt(np.mean(abs(dz[:n][k,ir])**2)):.4e}  rel diff vs eps={ref:g}: {rel(dz[:n][k,ir],dzr[:n][k,ir]):.3e}')
labs=list(D)
if len(labs)>=2:
    A,B=labs[0],labs[1]
    for e in sorted(set(D[A])&set(D[B])):
        for ir in ridxs:
            ta,dza=D[A][e]; tb,dzb=D[B][e]; tmax=min(ta[-1],tb[-1]); k=(ta>=tmin)&(ta<=tmax)
            zi=np.interp(ta[k],tb,dzb[:,ir].real)+1j*np.interp(ta[k],tb,dzb[:,ir].imag)
            print(f'[{A} vs {B}] eps={e:g} radius {ir}: rel L2 diff of dz over t in [{tmin},{tmax:.1f}] = {rel(dza[k,ir],zi):.3e}')
