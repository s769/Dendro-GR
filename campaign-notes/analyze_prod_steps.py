#!/usr/bin/env python3
import re,sys,statistics as st
f=sys.argv[1]
steps=[];events=[]
for l in open(f,errors='replace'):
    l=re.sub(r'\x1b\[[0-9;]*m','',l)
    m=re.match(r'(\d+\.\d+) (.*)',l)
    if not m: continue
    ts,rest=float(m.group(1)),m.group(2)
    s=re.search(r'Executing step\s*:\s*(\d+)\s*current time\s*:?([0-9.eE+-]+)\s*dt:([0-9.eE+-]+)',rest)
    if s: steps.append((int(s.group(1)),ts,float(s.group(2)),float(s.group(3)))); continue
    if re.search(r'Remesh is triggered|checkpoint|Checkpoint|extract-time|Writing|write_checkpt',rest): events.append((ts,rest.strip()[:110]))
print('steps logged',len(steps))
if len(steps)>2:
    d=[(steps[i][0],steps[i+1][1]-steps[i][1]) for i in range(len(steps)-1)]
    base=[x[1] for x in d]; med=st.median(base)
    print('median wall/step %.3f s   mean %.3f   total over logged steps %.1f s'%(med,st.mean(base),sum(base)))
    big=sorted(d,key=lambda x:-x[1])[:12]
    print('slowest steps (step, wall s):',[(a,round(b,2)) for a,b in big])
    print('excess over median in slow steps (>3x median): %.1f s of %.1f s (%.0f%%)'%(sum(b-med for a,b in d if b>3*med),sum(base),100*sum(b-med for a,b in d if b>3*med)/sum(base)))
    print('dt first/last: %.4g / %.4g, sim time %.3f -> %.3f'%(steps[0][3],steps[-1][3],steps[0][2],steps[-1][2]))
t0=steps[0][1] if steps else 0
for ts,r in events[:25]: print('  +%.1fs  %s'%(ts-t0,r))
