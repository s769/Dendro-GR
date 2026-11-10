import numpy as np
S=[[0,1,2],[1,3,4],[2,4,5]]
PAIRS=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
def make_env(N,rng):
    env={}
    g=np.eye(3)[None]+0.15*rng.standard_normal((N,3,3)); g=0.5*(g+g.transpose(0,2,1))
    for k,(i,j) in enumerate(PAIRS): env[f"gt{k}"]=g[:,i,j]
    A=0.3*rng.standard_normal((N,3,3)); A=0.5*(A+A.transpose(0,2,1))
    for k,(i,j) in enumerate(PAIRS): env[f"At{k}"]=A[:,i,j]
    env["K"]=0.2*rng.standard_normal(N); env["chi"]=0.3+0.7*rng.random(N)
    for i in range(3):
        env[f"grad_{i}_chi"]=rng.standard_normal(N)
        for k in range(6): env[f"grad_{i}_gt{k}"]=0.5*rng.standard_normal(N)
        for k in range(3): env[f"grad_{i}_Gt{k}"]=0.5*rng.standard_normal(N)
    for (i,j) in PAIRS:
        env[f"grad2_{i}_{j}_chi"]=rng.standard_normal(N)
        for k in range(6): env[f"grad2_{i}_{j}_gt{k}"]=0.5*rng.standard_normal(N)
    env["DENDRO_18"]=np.full(N,0.37)
    return env
def unpack(env,N):
    g=np.zeros((N,3,3)); At=np.zeros((N,3,3))
    for k,(i,j) in enumerate(PAIRS):
        g[:,i,j]=g[:,j,i]=env[f"gt{k}"]; At[:,i,j]=At[:,j,i]=env[f"At{k}"]
    dg=np.zeros((N,3,3,3))     # dg[:,l,i,j] = d_l g_ij
    ddg=np.zeros((N,3,3,3,3))  # ddg[:,l,m,i,j]
    for l in range(3):
        for k,(i,j) in enumerate(PAIRS): dg[:,l,i,j]=dg[:,l,j,i]=env[f"grad_{l}_gt{k}"]
    for (l,m) in PAIRS:
        for k,(i,j) in enumerate(PAIRS):
            v=env[f"grad2_{l}_{m}_gt{k}"]; ddg[:,l,m,i,j]=ddg[:,m,l,i,j]=ddg[:,l,m,j,i]=ddg[:,m,l,j,i]=v
    dGt=np.zeros((N,3,3))      # dGt[:,i,k] = d_i Gt_k
    for i in range(3):
        for k in range(3): dGt[:,i,k]=env[f"grad_{i}_Gt{k}"]
    chi=env["chi"]; dchi=np.stack([env[f"grad_{i}_chi"] for i in range(3)],1)
    ddchi=np.zeros((N,3,3))
    for (l,m) in PAIRS: ddchi[:,l,m]=ddchi[:,m,l]=env[f"grad2_{l}_{m}_chi"]
    return g,dg,ddg,dGt,chi,dchi,ddchi,At,env["K"]
def ham_literal(env,N):
    """literal transcription of CodeGen/dendro.py compute_ricci + bssn.py ham_computation"""
    g,dg,ddg,dGt,chi,dchi,ddchi,At,K=unpack(env,N)
    gi=np.linalg.inv(g)
    C1=np.zeros((N,3,3,3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                C1[:,k,i,j]=0.5*(dg[:,j,k,i]+dg[:,i,k,j]-dg[:,k,i,j])
    C2=np.einsum('nil,nljk->nijk',gi,C1)
    CalGt=np.einsum('nkl,nikl->ni',gi,C2)
    Rt=np.zeros((N,3,3))
    for i in range(3):
        for j in range(3):
            t=-0.5*np.einsum('nlm,nlm->n',gi,ddg[:,:,:,i,j])
            t=t+0.5*sum(g[:,k,i]*dGt[:,j,k]+g[:,k,j]*dGt[:,i,k] for k in range(3))
            t=t+0.5*sum(CalGt[:,k]*(C1[:,i,j,k]+C1[:,j,i,k]) for k in range(3))
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        t=t+gi[:,l,m]*(C2[:,k,l,i]*C1[:,j,k,m]+C2[:,k,l,j]*C1[:,i,k,m]+C2[:,k,i,m]*C1[:,k,l,j])
            Rt[:,i,j]=t
    xR=np.zeros((N,3,3)); Rphi=np.zeros((N,3,3))
    for i in range(3):
        for j in range(3):
            xR[:,i,j]=(1/(2*chi))*(ddchi[:,i,j]-sum(C2[:,k,j,i]*dchi[:,k] for k in range(3)))-1/(4*chi*chi)*dchi[:,i]*dchi[:,j]
    for i in range(3):
        for j in range(3):
            Rphi[:,i,j]=xR[:,i,j]+1/(2*chi)*g[:,i,j]*(
                sum(gi[:,k,l]*(ddchi[:,k,l]-3/(2*chi)*dchi[:,k]*dchi[:,l]) for k in range(3) for l in range(3))
                -sum(CalGt[:,m]*dchi[:,m] for m in range(3)))
    R=Rt+Rphi
    sqrA=np.einsum('nij,nik,njl,nkl->n',At,gi,gi,At)
    return np.einsum('n,njk,njk->n',chi,gi,R)-sqrA+(2/3)*K**2
if __name__=="__main__":
    import ham_check as hc
    rng=np.random.default_rng(1); N=2000
    env=make_env(N,rng)
    B=hc.eval_generated(env)
    H=ham_literal(env,N)
    r=B/(-12*H)
    print("generated B vs -12*H_literal: ratio min %.12f max %.12f"%(r.min(),r.max()))
    print("max rel err |B+12H|/|B|: %.3e"%np.max(np.abs(B+12*H)/np.maximum(np.abs(B),1e-300)))
