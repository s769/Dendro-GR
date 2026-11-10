import numpy as np,sys,glob,os
R=os.path.expanduser('~/Dendro-GR-runs/'+os.environ.get('SC','statecmp')+(('_'+sys.argv[1]) if len(sys.argv)>1 else ''))
names="alpha chi K gt0 gt1 gt2 gt3 gt4 gt5 beta0 beta1 beta2 At0 At1 At2 At3 At4 At5 Gt0 Gt1 Gt2 B0 B1 B2".split()
def rd(f):
    with open(f,'rb') as fh:
        n,nv=np.fromfile(fh,dtype=np.uint64,count=2); a=np.fromfile(fh,dtype=np.float64).reshape(int(nv),int(n)); return a
for step in ((1,) if len(sys.argv)>1 else (1,2,5)):
    print("== after",step,"steps")
    for r in range(8):
        fc=f"{R}/cpu/dump/state_step{step}_rank{r}.bin"; fg=f"{R}/gpu/dump/state_step{step}_rank{r}.bin"
        if not(os.path.exists(fc) and os.path.exists(fg)): continue
        c,g=rd(fc),rd(fg)
        if c.shape!=g.shape: print(" rank",r,"shape mismatch",c.shape,g.shape); continue
        if r==0:
            for v in range(c.shape[0]):
                d=np.abs(c[v]-g[v]); sc=max(np.abs(c[v]).max(),1e-300); i=d.argmax()
                print(f" {names[v] if v<len(names) else v:6s} max|d|={d.max():.3e} rel-to-max={d.max()/sc:.2e} at node {i} nnz>1e-13: {(d>1e-13*sc).sum()}/{d.size}")
        else:
            print(" rank",r,"max rel diff over vars %.2e"%max(np.abs(c[v]-g[v]).max()/max(np.abs(c[v]).max(),1e-300) for v in range(c.shape[0])))
