#!/usr/bin/env python3
"""Accuracy-vs-cost: compare psi4_22 (phase, amplitude) of resolution/tolerance variants against the finest run.
usage: analyze_res.py TEND REF_NAME NAME1 NAME2 ...   (dirs ~/Dendro-GR-runs/resstudy/<name>_T<TEND>)"""
import sys,os,re,numpy as np
T=sys.argv[1]; ref=sys.argv[2]; names=sys.argv[3:]
R=os.path.expanduser("~/Dendro-GR-runs/resstudy")
def load(name,ridx):
    f=f"{R}/{name}_T{T}/dat/bssn_profile_GW_l2_m2.dat"
    t=[];z=[]
    for l in open(f):
        if l.startswith(("Time","#")) or not l.strip(): continue
        p=l.split(); m=re.findall(r"\(([^,]+),([^)]+)\)",l)
        t.append(float(p[1])); z.append(complex(float(m[ridx][0]),float(m[ridx][1])))
    return np.array(t),np.array(z)
def cost(name):
    log=f"{R}/{name}_T{T}/run.log"; L=re.sub(r"\x1b\[[0-9;]*m","",open(log).read())
    steps=len(re.findall(r"Executing step",L)); 
    m=re.findall(r"ETS time \(max\) : ([0-9.]+)",L); 
    last=re.findall(r"Executing step :\s+(\d+)",L)
    return (float(m[-1]) if m else float('nan')), (int(last[-1]) if last else 0)
print("cost (stepping-phase ETS wall, last step):")
for n in [ref]+names: print("  %-8s ETS=%8.1f s  last step %s"%(n,*cost(n)))
for ridx,rname in ((5,"r=25"),(3,"r=18")):
    tr,zr=load(ref,ridx); ph_r=np.unwrap(np.angle(zr)); A_r=np.abs(zr)
    print(f"\n=== {rname}: error vs reference '{ref}' on common times ===")
    res={}
    for n in names:
        t,z=load(n,ridx); ph=np.unwrap(np.angle(z)); A=np.abs(z)
        idx=[(i,int(np.argmin(np.abs(tr-ti)))) for i,ti in enumerate(t) if np.min(np.abs(tr-ti))<1e-6]
        if not idx: print("  %-8s no common times"%n); continue
        ii=np.array([a for a,b in idx]); jj=np.array([b for a,b in idx])
        # phase offset removed at the first common sample after the signal has arrived (|z| above 10% of max)
        sel=A_r[jj]>0.1*A_r.max(); k0=np.where(sel)[0][0] if sel.any() else 0
        dph=(ph[ii]-ph_r[jj]); dph-=dph[k0]
        dA=(A[ii]-A_r[jj])/np.maximum(A_r[jj],1e-300)
        mask=sel
        print("  %-8s samples %3d | max|dphase| %.3e rad | final dphase %.3e rad | max|dA/A| %.3e | rms|dA/A| %.3e"%(n,len(ii),np.max(np.abs(dph[mask])),dph[-1],np.max(np.abs(dA[mask])),np.sqrt(np.mean(dA[mask]**2))))
        res[n]=(np.max(np.abs(dph[mask])),np.max(np.abs(dA[mask])))
    if len(names)>=2 and all(n in res for n in names[:2]):
        a,b=res[names[0]],res[names[1]]
        print("  error ratio %s/%s: phase %.2f, amplitude %.2f  (-> observed order ~ log2 of that if one level apart)"%(names[0],names[1],a[0]/max(b[0],1e-300),a[1]/max(b[1],1e-300)))
