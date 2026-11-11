#!/usr/bin/env python3
"""Eccentricity estimate from the BH tracks (dgr_BHLocations.dat): separation r(t) and orbital phase.
usage: estimate_ecc.py RUNDIR [tmin tmax] -- e_r ~ amplitude of the oscillation of r(t) about a smooth polynomial trend / r,
and e_Omega from the oscillation of Omega = dphi/dt about its trend (Omega ~ r^-3/2 so e_Omega ~ 2/3 * amplitude/Omega for a Newtonian orbit...).
Both are crude (valid for e << 1); they track the *residual* eccentricity of the initial data."""
import sys,numpy as np
d=sys.argv[1]; tmin=float(sys.argv[2]) if len(sys.argv)>2 else 15; tmax=float(sys.argv[3]) if len(sys.argv)>3 else 70
a=np.loadtxt(d+"/dat/dgr_BHLocations.dat",skiprows=1)
t=a[:,1]; dx=a[:,2]-a[:,5]; dy=a[:,3]-a[:,6]
r=np.hypot(dx,dy); ph=np.unwrap(np.arctan2(dy,dx))
k=(t>=tmin)&(t<=tmax); tt=t[k]; x=(tt-tt.mean())/(tt.max()-tt.min())
def osc(y,deg=4):
    c=np.polyfit(x,y[k],deg); res=y[k]-np.polyval(c,x); return res,np.polyval(c,x)
resr,rt=osc(r); er=(resr.max()-resr.min())/2/rt.mean()
Om=np.gradient(ph,t); resO,Ot=osc(Om); eO=(2.0/3.0)*((resO.max()-resO.min())/2)/Ot.mean()
# frequency of the oscillation (for sanity): dominant FFT peak of resr
n=len(tt); f=np.fft.rfftfreq(n,np.mean(np.diff(tt))); F=abs(np.fft.rfft(resr-resr.mean())); fp=f[1:][np.argmax(F[1:])] if n>4 else 0
print(f"{d.split('Dendro-GR-runs/')[-1]}: t in [{tmin},{tmax}] r~{rt.mean():.3f} (trend {rt[0]:.3f}->{rt[-1]:.3f}) e_r={er:.4f} e_Omega={eO:.4f} osc.period~{1/fp if fp>0 else float('nan'):.1f}  Omega~{Ot.mean():.4f} (orbital period {2*np.pi/Ot.mean():.1f})")
