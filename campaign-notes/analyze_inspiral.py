#!/usr/bin/env python3
"""Inspiral-window comparison of psi4_22: python3 analyze_inspiral.py REF VAR [ridx=0] [u0=60]  (dirs relative to ~/Dendro-GR-runs)
Uses retarded time u=t-r_ex (r_ex from the GW radii 50..100 by index) and only u>=u0 (after the junk transient). Prints the unwrapped phase
difference and relative amplitude difference at several u, the maximum over the window, and the u of the |psi4| peak (merger) in REF."""
import re,sys,numpy as np
R="/home1/08435/srvenkat/Dendro-GR-runs/"; RAD=[50,60,70,80,90,100]
ref,var=sys.argv[1],sys.argv[2]; ri=int(sys.argv[3]) if len(sys.argv)>3 else 0; u0=float(sys.argv[4]) if len(sys.argv)>4 else 60.
def load(p):
    t,z=[],[]
    for l in open(R+p+"/dat/dgr_GW_l2_m2.dat").read().splitlines()[1:]:
        q=l.split(); m=re.findall(r"\(([^,]+),([^)]+)\)",l); t.append(float(q[1])); z.append(complex(float(m[ri][0]),float(m[ri][1])))
    return np.array(t)-RAD[ri],np.array(z)
ur,zr=load(ref); uv,zv=load(var)
ue=min(ur[-1],uv[-1]); x=ur[(ur>=u0)&(ur<=ue)]
pr=np.interp(x,ur,np.unwrap(np.angle(zr))); pv=np.interp(x,uv,np.unwrap(np.angle(zv)))
ar=np.interp(x,ur,abs(zr)); av=np.interp(x,uv,abs(zv)); d=pv-pr; d-=d[0]   # remove the arbitrary offset at the window start
k=np.argmax(abs(zr)); print(f"ref peak |psi4|={abs(zr).max():.3e} at u={ur[k]:.1f}; window u in [{x[0]:.0f},{x[-1]:.0f}] (var ends at u={uv[-1]:.0f}, ref u={ur[-1]:.0f})")
for uu in (u0,u0+40,u0+80,u0+120,u0+160,ue-10,ue):
    if uu<=x[-1]: i=np.argmin(abs(x-uu)); print(f"  u={x[i]:6.1f}  dphase={d[i]:+.4f} rad  d|A|/|A|={(av[i]/ar[i]-1):+.4f}  |psi4|_ref={ar[i]:.2e}")
print(f"  max|dphase|={abs(d).max():.4f} rad   max|dA/A|={abs(av/ar-1).max():.4f}")
