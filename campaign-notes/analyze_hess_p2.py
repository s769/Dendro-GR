import re,numpy as np
O="/home1/08435/srvenkat/Dendro-GR-runs/fdtp_c13/"
def load(tag):
    t,z=[],[]
    for l in open(O+tag+"/dat/dgr_GW_l2_m2.dat").read().splitlines()[1:]:
        p=l.split(); t.append(float(p[1])); z.append(complex(*map(float,re.findall(r"[-+0-9.eE]+",p[4]))))
    return np.array(t),np.array(z)
Z={k:load(k)[1] for k in ["fid","p_0.001","m_0.001","b2p","b2m","pp","mm","pm","mp"]}; t=load("fid")[0]
n=min(len(v) for v in Z.values()); e=.001; f=Z["fid"][:n]
g=lambda k:Z[k][:n]
J1=(g("p_0.001")-g("m_0.001"))/(2*e); J2=(g("b2p")-g("b2m"))/(2*e)
H11=(g("p_0.001")+g("m_0.001")-2*f)/e**2; H22=(g("b2p")+g("b2m")-2*f)/e**2
H12=(g("pp")+g("mm")-g("pm")-g("mp"))/(4*e**2)
print("t   |z|   |J1|/|z| |J2|/|z| J1-J2 sym | |H11| |H22| |H12| (rel) | H11-H22 sym | total-mass check: (H11+H22+2H12) ~ d2/dM^2")
for i in (20,40,60,70,80,100,140,190,n-1):
    a=abs(f[i])
    print(f"{t[i]:6.1f} {a:.1e} {abs(J1[i])/a:.3e} {abs(J2[i])/a:.3e} {abs(J1[i]-J2[i])/abs(J1[i]):.1e} | {abs(H11[i])/a:.3e} {abs(H22[i])/a:.3e} {abs(H12[i])/a:.3e} | {abs(H11[i]-H22[i])/abs(H11[i]):.1e}")
# total-mass direction: scaling M -> lambda M, psi4 is dimensionally scaled; check dz/dm1 + dz/dm2 vs analytic scale derivative
