import re,numpy as np
O="/home1/08435/srvenkat/Dendro-GR-runs/fdtp_c13/"
def load(tag):
    t,z=[],[]
    for l in open(O+tag+"/dat/dgr_GW_l2_m2.dat").read().splitlines()[1:]:
        p=l.split(); t.append(float(p[1])); z.append(complex(*map(float,re.findall(r"[-+0-9.eE]+",p[4]))))
    return np.array(t),np.array(z)
tags=["fid","p_0.001","m_0.001","pp","mm","pm","mp","p_0.0003","m_0.0003","pp3","mm3","pm3","mp3","p_0.003","m_0.003"]
Z={k:load(k)[1] for k in tags}; t=load("fid")[0]; n=min(len(v) for v in Z.values()); g=lambda k:Z[k][:n]; f=g("fid")
def stencil(e,p,m,pp,mm,pm,mp):
    return (g(p)-g(m))/(2*e), (g(p)+g(m)-2*f)/e**2, (g(pp)+g(mm)-g(pm)-g(mp))/(4*e**2)
A=stencil(.001,"p_0.001","m_0.001","pp","mm","pm","mp"); B=stencil(.0003,"p_0.0003","m_0.0003","pp3","mm3","pm3","mp3")
J3=(g("p_0.003")-g("m_0.003"))/.006; H3=(g("p_0.003")+g("m_0.003")-2*f)/.003**2
print("plateau |X(1e-3)-X(3e-4)|/|X(3e-4)|  [J1, H11, H12]   and eps=3e-3 vs 1e-3 for J1,H11")
for i in (20,40,60,70,80,100,140,190,n-1):
    r=lambda a,b: abs(a[i]-b[i])/abs(b[i])
    print(f"t={t[i]:6.1f} {r(A[0],B[0]):.1e} {r(A[1],B[1]):.1e} {r(A[2],B[2]):.1e} | {abs(J3[i]-A[0][i])/abs(A[0][i]):.1e} {abs(H3[i]-A[1][i])/abs(A[1][i]):.1e}")
