"""Plan-level combo search for 2_medium: from a base solution, find combos of <=3 role changes
(off/full/half) with value gain > 0, energy within cap, and exact multi-threshold free-data consumption
C_t <= F_t - margin_t. Sources/sourced reqs kept as in base. Writes candidate plans (best first)."""
import sys, json, itertools, time
from collections import defaultdict
import numpy as np
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']; cap=inst['energyCap']
TY='ntic'; nM=len(M)
S=json.load(open(sys.argv[1])); margin=[int(x) for x in sys.argv[2].split(',')]; out=sys.argv[3]
maxdv=int(sys.argv[4]) if len(sys.argv)>4 else 40
srcs={s for s,_ in S['modelMappings']}; tg=defaultdict(set)
for s,t in S['modelMappings']: tg[t].add(next(iter(M[s]['dataRequirements'])))
cp=defaultdict(int); used=set()
for dd,m in S['dataMappings']: used.add(m); cp[m]|=D[dd]['isCopyrighted']
used|=set(tg)
role={}
for i in range(nM):
    if i in srcs: role[i]='S'
    elif i in used: role[i]='H' if (i in tg or cp[i]) else 'F'
    else: role[i]='O'
fixed={i for i in range(nM) if role[i]=='S' or i in tg}
free={t:sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted']) for t in TY}
Ft={t:sum(free[t]) for t in TY}
P={t:np.concatenate([[0],np.cumsum(free[t])]) for t in TY}
bad=[any(r['upperBound']<r['lowerBound'] for r in m['dataRequirements'].values()) for m in M]
def C(fullset,t):
    lbs=sorted(M[i]['dataRequirements'][t]['lowerBound'] for i in fullset if t in M[i]['dataRequirements'])
    a=np.array(lbs,dtype=np.int64); n=len(a)
    if n==0: return 0
    suf=np.concatenate([np.cumsum(a[::-1])[::-1],[0]]); pre=np.concatenate([[0],np.cumsum(a)])
    j=np.arange(n+1); Pj=np.where(j<len(P[t]),P[t][np.minimum(j,len(P[t])-1)],10**15)
    return int(np.max(suf+np.maximum(Pj,pre)))
full0={i for i in range(nM) if role[i]=='F'}
val0=sum(M[i]['value'] for i in full0)+sum(M[i]['value']//2 for i in range(nM) if role[i]=='H')
E0=sum(M[i]['energyCost'] for i in range(nM) if role[i]!='O')
sl0={t:Ft[t]-margin[TY.index(t)]-C(full0,t) for t in TY}
print('base value',val0,'Eslack',cap-E0,'free slack-margin',sl0,flush=True)
# moves
mv=[]  # (i,newrole,dv,dE,dvec)
def vv(i,r): return 0 if r=='O' else (M[i]['value'] if r=='F' else M[i]['value']//2)
for i in range(nM):
    if i in fixed: continue
    for r in 'OFH':
        if r==role[i]: continue
        if r=='F' and bad[i]: continue
        if r=='H' and bad[i]: continue  # bad needs sourcing; skip
        dv=vv(i,r)-vv(i,role[i]); dE=(0 if r=='O' else M[i]['energyCost'])-(0 if role[i]=='O' else M[i]['energyCost'])
        dvec=[0]*4
        for t,q in M[i]['dataRequirements'].items():
            k=TY.index(t)
            if r=='F': dvec[k]+=q['lowerBound']
            if role[i]=='F': dvec[k]-=q['lowerBound']
        mv.append((i,r,dv,dE,dvec))
print('moves',len(mv),flush=True)
I_=np.array([m[0] for m in mv]); DV=np.array([m[2] for m in mv],dtype=np.int64); DE=np.array([m[3] for m in mv],dtype=np.int64)
DD=np.array([m[4] for m in mv],dtype=np.int64)
Es=cap-E0; sl=np.array([sl0[t] for t in TY],dtype=np.int64)
found=[]
def exact_ok(combo):
    f=set(full0); 
    for k in combo:
        i,r=mv[k][0],mv[k][1]
        f.discard(i)
        if r=='F': f.add(i)
    ok=all(C(f,t)<=Ft[t]-margin[TY.index(t)] for t in TY)
    return ok
t0=time.time()
# singles & pairs (vectorized)
n=len(mv)
for a in range(n):
    if DV[a]<=-maxdv: continue
    b=np.arange(a+1,n)
    dv=DV[a]+DV[b]; dE=DE[a]+DE[b]; dd=DD[a][None,:]+DD[b]
    m=(dv>0)&(dE<=Es)&np.all(dd<=sl[None,:],axis=1)&(I_[b]!=I_[a])
    for bb in b[m]:
        found.append((int(DV[a]+DV[bb]),(a,int(bb))))
print('pairs found',len(found),time.time()-t0,flush=True)
found.sort(reverse=True)
res=[]; 
for g,combo in found[:3000]:
    if exact_ok(combo): res.append((g,combo))
    if len(res)>=50: break
print('exact-ok pairs',[(g,[(mv[k][0],mv[k][1]) for k in c]) for g,c in res[:10]],flush=True)
plans=[]
for g,combo in res:
    rr=dict(role)
    for k in combo: rr[mv[k][0]]=mv[k][1]
    p={'full':[i for i in range(nM) if rr[i]=='F'],'half':[i for i in range(nM) if rr[i]=='H'],'source':sorted(srcs),
       'srcreq':[[i,t] for i in tg for t in tg[i]],'obj':val0+g}
    plans.append(p)
json.dump(plans,open(out,'w')); print('wrote',len(plans),'plans',flush=True)
