import numpy as np, sys
sys.path.insert(0,'.')
import ham_hand as hh, ham_check as hc
S=hh.S; PAIRS=hh.PAIRS
def ham_contracted(env,N):
    g=[env[f"gt{k}"] for k in range(6)]
    At=[env[f"At{k}"] for k in range(6)]; K=env["K"]; chi=env["chi"]
    # inverse metric by adjugate / det (same structure as the generator)
    g0,g1,g2,g3,g4,g5=g
    det=g0*(g3*g5-g4*g4)-g1*(g1*g5-g4*g2)+g2*(g1*g4-g3*g2)
    gi6=[(g3*g5-g4*g4)/det,(g2*g4-g1*g5)/det,(g1*g4-g2*g3)/det,(g0*g5-g2*g2)/det,(g1*g2-g0*g4)/det,(g0*g3-g1*g1)/det]
    gi=lambda i,j: gi6[S[i][j]]
    dg=[[env[f"grad_{l}_gt{k}"] for k in range(6)] for l in range(3)]          # dg[l][k]
    # second derivative operator on each g component: Tg[k] = gi_lm d_l d_m g_k
    def d2(l,m,k): return env[f"grad2_{min(l,m)}_{max(l,m)}_gt{k}"]
    Tg=[sum(gi(l,m)*d2(l,m,k) for l in range(3) for m in range(3)) for k in range(6)]
    mult=[1,2,2,1,2,1]
    S1=-0.5*sum(mult[k]*gi6[k]*Tg[k] for k in range(6))
    divG=env["grad_0_Gt0"]+env["grad_1_Gt1"]+env["grad_2_Gt2"]
    # Christoffels
    C1=[[None]*6 for _ in range(3)]
    for a in range(3):
        for s,(b,c) in enumerate(PAIRS):
            C1[a][s]=0.5*(dg[c][S[a][b]]+dg[b][S[a][c]]-dg[a][S[b][c]])
    C2=[[sum(gi(a,l)*C1[l][s] for l in range(3)) for s in range(6)] for a in range(3)]
    CalGt=[sum(gi(b,c)*C2[a][S[b][c]] for b in range(3) for c in range(3)) for a in range(3)]
    Lam=[sum(gi(i,j)*dg[k][S[i][j]] for i in range(3) for j in range(3)) for k in range(3)]
    t3=sum(CalGt[k]*0.5*Lam[k] for k in range(3))
    t4a=2*sum(gi(l,m)*C2[k][S[l][i]]*C2[i][S[k][m]] for k in range(3) for l in range(3) for m in range(3) for i in range(3))
    E=[[[sum(gi(i,j)*C1[k][S[l][j]] for j in range(3)) for i in range(3)] for l in range(3)] for k in range(3)]  # E[k][l][i]
    t4b=sum(gi(l,m)*C2[k][S[i][m]]*E[k][l][i] for k in range(3) for l in range(3) for m in range(3) for i in range(3))
    SRt=S1+divG+t3+t4a+t4b
    dchi=[env[f"grad_{i}_chi"] for i in range(3)]
    def d2c(l,m): return env[f"grad2_{min(l,m)}_{max(l,m)}_chi"]
    Lc=sum(gi(l,m)*d2c(l,m) for l in range(3) for m in range(3))
    Q=sum(gi(l,m)*dchi[l]*dchi[m] for l in range(3) for m in range(3))
    cg=sum(CalGt[m]*dchi[m] for m in range(3))
    SRphi=(2/chi)*(Lc-cg)-(5/(2*chi*chi))*Q
    M=[[sum(gi(i,k)*At[S[k][j]] for k in range(3)) for j in range(3)] for i in range(3)]   # gi*At
    AA=sum(M[i][j]*M[j][i] for i in range(3) for j in range(3))
    return chi*(SRt+SRphi)-AA+(2/3)*K**2
if __name__=="__main__":
    rng=np.random.default_rng(7); N=3000
    env=hh.make_env(N,rng)
    B=hc.eval_generated(env); Hc=ham_contracted(env,N); Hl=hh.ham_literal(env,N)
    print("contracted vs literal:   max rel err %.3e"%np.max(np.abs(Hc-Hl)/np.maximum(np.abs(Hl),1e-300)))
    print("contracted vs generated: max rel err of B=-12H: %.3e"%np.max(np.abs(B+12*Hc)/np.maximum(np.abs(B),1e-300)))
