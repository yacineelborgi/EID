"""Decompose a medium solution's value against the certified multi-threshold LP duals: B - value = col devs + row slacks."""
import sys, json, pickle
from collections import defaultdict
import numpy as np
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']
C=pickle.load(open(sys.argv[2],'rb')); ye=[float(v) for v in C['ye']]; rcs=[float(v) for v in C['rcs']]; B=float(C['B'])
TH=tuple(sorted(set(int(x) for x in sys.argv[3].split(',')))) if len(sys.argv)>3 else mip6.THETAS
mip6.SC=1.0
mo=mip6.build(inst,exact=True,thetas=TH)
S=json.load(open(sys.argv[1]))
srcs={s for s,_ in S['modelMappings']}; tg=defaultdict(set)
for s,t in S['modelMappings']: tg[t].add(next(iter(M[s]['dataRequirements'])))
cp=defaultdict(int); used=set()
for dd,m in S['dataMappings']: used.add(m); cp[m]|=D[dd]['isCopyrighted']
used|=set(tg)
x=np.zeros(mo.nTot)
for i in used-srcs:
    if i in tg or cp[i]: x[mo.ho+i]=1
    else: x[mo.fo+i]=1
for i in srcs: x[mo.so+i]=1
for k,(i,t,L,isb) in enumerate(mo.zlist):
    if i in tg and t in tg[i]: x[mo.zo+k]=1
A=mo.A.tocsr()
# continuous: P rows 'P_*' : P - sum = 0 ; K rows: K - sum = 0 ; W = S(K)
for pool,t,th,K,W in mo.fam:
    its=mo.fs[t] if pool=='F' else mo.al[t]
    if th is None:
        r=mo.names.index(f'P_{pool}_{t}'); row=A.getrow(r); x[K]=-(row.toarray()[0][:mo.nV]@x[:mo.nV])
    else:
        r=mo.names.index(f'K_{pool}_{t}_{th}'); row=A.getrow(r); k=int(round(-(row.toarray()[0][:mo.nV]@x[:mo.nV]))); x[K]=k
        x[W]=sum(its[:k])
Ax=A@x
val=mo.cost@x
coldev=0; cd=[]
for j in range(mo.nTot):
    rc=rcs[j]
    if rc>0: d=rc*(mo.ub[j]-x[j])
    elif rc<0: d=rc*(mo.lb[j]-x[j])
    else: d=0
    if abs(d)>1e-9: cd.append((d,j)); coldev+=d
rows=defaultdict(float); viol=[]
for r,nm in enumerate(mo.names):
    y=ye[r]
    if y>0: s=y*(mo.rhi[r]-Ax[r]); 
    elif y<0: s=y*(mo.rlo[r]-Ax[r])
    else: s=0
    if s<-1e-6: viol.append((nm,s))
    key=nm if nm in('energy',) else '_'.join(nm.split('_')[:3]) 
    rows[key]+=s
print('value',val,'B',B,'B-val',B-val,'coldev',coldev,'rowslack',sum(rows.values()))
print('violated rows (should be none)',viol[:5])
for k,v in sorted(rows.items(),key=lambda kv:-abs(kv[1])):
    if abs(v)>0.01: print('  ',k,round(v,3))
cd.sort(reverse=True); print('top col devs',[(round(d,3),j) for d,j in cd[:12]])
