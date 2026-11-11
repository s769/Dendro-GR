import re,numpy as np,sys
R="/home1/08435/srvenkat/Dendro-GR-runs/"
def load(p,ri=5):
    t,z=[],[]
    for l in open(R+p+"/dat/dgr_GW_l2_m2.dat").read().splitlines()[1:]:
        q=l.split(); m=re.findall(r"\(([^,]+),([^)]+)\)",l)
        t.append(float(q[1])); z.append(complex(float(m[ri][0]),float(m[ri][1])))
    return np.array(t),np.array(z)
def cmp(ref,var,ri=5):
    tr,zr=load(ref,ri); amax=abs(zr).max(); t,z=load(var,ri); x=tr[(tr>20)&(tr<=min(t[-1],tr[-1]))]
    b=np.interp(x,t,abs(z))*np.exp(1j*np.interp(x,t,np.unwrap(np.angle(z)))); a=np.interp(x,tr,abs(zr))*np.exp(1j*np.interp(x,tr,np.unwrap(np.angle(zr))))
    d=np.unwrap(np.angle(b))-np.unwrap(np.angle(a)); mk=abs(a)>0.1*amax
    return f"{var} vs {ref}: max|dphase| {abs(d[mk]).max():.4f} max|dz|/max|z| {abs(b-a).max()/amax:.4f} (|z|max {amax:.2e}, t_peak {tr[np.argmax(abs(zr))]:.0f}, t_end {min(t[-1],tr[-1]):.0f})"
for a in sys.argv[1:]:
    ref,var=a.split(",");print(cmp(ref,var))
