#!/usr/bin/env python3
"""Compare runs: python3 compare_phys.py REF=dir OTHER=dir ... (dirs under ~/Dendro-GR-runs, files bssn_profile_*.dat or dat/ ...)."""
import sys,re,os,numpy as np
R=os.path.expanduser('~/Dendro-GR-runs')
def table(d,name):
    for sub in ['','dat/']:
        f=f'{R}/{d}/{sub}bssn_profile_{name}.dat'
        if os.path.exists(f): break
    else: return None,None
    t=[];v=[]
    for l in open(f).read().splitlines():
        p=l.split()
        if not p or not re.match(r'^[0-9]+$',p[0]): continue
        try: t.append(float(p[1])); v.append([complex(*map(float,re.findall(r'[-+0-9.eE]+',c))) if '(' in c else float(c) for c in p[2:]])
        except: pass
    return np.array(t),np.array(v)
def interp(t0,t1,v1):
    return np.array([np.interp(t0,t1,v1[:,k].real)+1j*np.interp(t0,t1,v1[:,k].imag) for k in range(v1.shape[1])]).T
args=dict(a.split('=') for a in sys.argv[1:]); ref=list(args)[0]
for name,desc in [('BHLocations','BH positions'),('Constraints','constraints'),('GW_l2_m2','psi4 l2m2'),('GW_l3_m2','psi4 l3m2')]:
    t0,v0=table(args[ref],name)
    if t0 is None: print(name,'missing in ref'); continue
    for lab,d in list(args.items())[1:]:
        t,v=table(d,name)
        if t is None: print(f'{desc:14s} {lab}: missing'); continue
        T=min(t0[-1],t[-1]); m=t0<=T+1e-9
        a=v0[m]; b=interp(t0[m],t,v.astype(complex)); 
        if name=='BHLocations':
            a=np.real(a); b=np.real(b)
            sep0=np.hypot(a[:,0]-a[:,3],a[:,1]-a[:,4]); sep1=np.hypot(b[:,0]-b[:,3],b[:,1]-b[:,4])
            print(f'{desc:14s} {lab} vs {ref}: t<= {T:.2f}  max|d pos| {np.max(abs(a-b)):.3e}  max|d sep| {np.max(abs(sep0-sep1)):.3e} (sep {sep0[-1]:.5f} vs {sep1[-1]:.5f})')
        elif name=='Constraints':
            a=np.real(a); b=np.real(b)
            print(f'{desc:14s} {lab} vs {ref}: C_HAM final {a[-1,0]:.4e} vs {b[-1,0]:.4e}; max C_HAM {a[:,0].max():.3e} vs {b[:,0].max():.3e}; max rel diff C_HAM {np.max(abs(a[:,0]-b[:,0])/np.maximum(abs(a[:,0]),1e-30)):.2e}')
        else:
            d_=abs(a-b); print(f'{desc:14s} {lab} vs {ref}: max|dz| {d_.max():.3e} (|ref|max {abs(a).max():.3e}) rel-to-max {d_.max()/abs(a).max():.2e}; final-row rel diff {np.max(abs(a[-1]-b[-1])/np.maximum(abs(a[-1]),1e-30)):.2e}')
