import numpy as np,os,sys
R=os.path.expanduser("~/Dendro-GR-runs/"+os.environ.get("SC","statecmp"))
names="alpha chi K gt0 gt1 gt2 gt3 gt4 gt5 beta0 beta1 beta2 At0 At1 At2 At3 At4 At5 Gt0 Gt1 Gt2 B0 B1 B2".split()
def rd(f):
    with open(f,'rb') as fh:
        usz,nv,nb=[int(x) for x in np.fromfile(fh,dtype=np.uint64,count=3)]
        blk=np.fromfile(fh,dtype=np.uint64,count=5*nb).reshape(nb,5).astype(int)
        d=np.fromfile(fh,dtype=np.float64).reshape(nv,usz); return blk,d
step=int(sys.argv[1]) if len(sys.argv)>1 else 1
PW=3
for r in range(8):
    fc=f"{R}/cpu/dump/unzip_step{step}_rank{r}.bin"; fg=f"{R}/gpu/dump/unzip_step{step}_rank{r}.bin"
    if not (os.path.exists(fc) and os.path.exists(fg)): continue
    bc,c=rd(fc); bg,g=rd(fg)
    if not (bc==bg).all(): print("rank",r,"block tables differ"); continue
    tot_int=0; tot_pad=0; worst=[]
    for b,(off,nx,ny,nz,fl) in enumerate(bc):
        n=nx*ny*nz
        for v in (0,2,3,9,12,18):
            a=c[v,off:off+n].reshape(nz,ny,nx); q=g[v,off:off+n].reshape(nz,ny,nx)
            d=np.abs(a-q)
            inter=d[PW:nz-PW,PW:ny-PW,PW:nx-PW]
            pad=d.copy(); pad[PW:nz-PW,PW:ny-PW,PW:nx-PW]=0
            worst.append((inter.max(),pad.max(),b,v,fl))
    wi=max(worst,key=lambda t:t[0]); wp=max(worst,key=lambda t:t[1])
    print(f"rank {r}: blocks={len(bc)} max|diff| interior={wi[0]:.2e} (blk {wi[2]} var {names[wi[3]]}) padding={wp[1]:.2e} (blk {wp[2]} var {names[wp[3]]} bflag {wp[4]})")
if step>=1000:
    # per-variable interior diff of the RHS output on rank 0, for ko-sensitive diagnosis
    bc,c=rd(f"{R}/cpu/dump/unzip_step{step}_rank0.bin"); bg,g=rd(f"{R}/gpu/dump/unzip_step{step}_rank0.bin")
    print("RHS output (stage 1, step",step-1000,") per var, rank0, interior:")
    for v in range(24):
        mx=0; sc=0; wb=-1
        for b,(off,nx,ny,nz,fl) in enumerate(bc):
            n=nx*ny*nz
            a=c[v,off:off+n].reshape(nz,ny,nx)[PW:nz-PW,PW:ny-PW,PW:nx-PW]; q=g[v,off:off+n].reshape(nz,ny,nx)[PW:nz-PW,PW:ny-PW,PW:nx-PW]
            d=np.abs(a-q).max(); sc=max(sc,np.abs(a).max())
            if d>mx: mx=d; wb=b
        print(f"  {names[v]:6s} max|d|={mx:.3e}  max|rhs|={sc:.3e}  rel={mx/max(sc,1e-300):.2e} blk {wb}")
