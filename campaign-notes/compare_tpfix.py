#!/usr/bin/env python3
"""Effect of reusing the fiducial TwoPunctures solution for a perturbed mass on the FD sensitivity.
dz_old = z(p_0.001 [reused fid TP file]) - z(fid);  dz_fix = z(p_0.001_tpfix [correctly solved]) - z(fid)."""
import re,glob,os,numpy as np
R=os.path.expanduser("~/Dendro-GR-runs/tier15_wave_coarse_T26_cahd")
def load(name):
    f=glob.glob(f"{R}/{name}/dat/*_GW_l2_m2.dat")[0]; t=[];z=[]
    for l in open(f):
        if l.startswith(("Time","#")) or not l.strip(): continue
        p=l.split(); m=re.findall(r"\(([^,]+),([^)]+)\)",l)
        t.append(float(p[1])); z.append([complex(float(a),float(b)) for a,b in m])
    return np.array(t),np.array(z)
tf,zf=load("fid"); to,zo=load("p_0.001"); tx,zx=load("p_0.001_tpfix")
n=min(len(tf),len(to),len(tx)); print("samples",n,"t_end %.2f"%tf[n-1])
zf,zo,zx=zf[:n],zo[:n],zx[:n]
for ridx,r in enumerate([10,12,15,18,20,25]):
    dzo=zo[:,ridx]-zf[:,ridx]; dzx=zx[:,ridx]-zf[:,ridx]
    nz=np.linalg.norm(dzo)
    print("r=%2d: ||dz_old||=%.3e  ||dz_fix||=%.3e  ||dz_fix-dz_old||/||dz_old|| = %.3e   (relative change of the sensitivity if the TP solution is re-solved)"%(r,nz,np.linalg.norm(dzx),np.linalg.norm(dzx-dzo)/max(nz,1e-300)))
