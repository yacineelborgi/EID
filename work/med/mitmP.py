"""MITM plan search for 2_medium around a base solution.
Moves = role changes (off/full/half) of non-fixed models with reduced value r=dv-uE dE-uF.dd >= -Rm.
Subsets (<=K moves, distinct models) on two model halves; KD-tree box match so that leftover slack after the
combo lies in [mt, mt+Wt] per free type and [0, WE] for energy. Exact multi-threshold check, write plans.
usage: mitmM.py base_sol out_plans Rm Rtot K mt Wt WE seed"""
import sys, json, time, itertools, random
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
base=sys.argv[1]; out=sys.argv[2]; Rm=float(sys.argv[3]); Rtot=float(sys.argv[4]); K=int(sys.argv[5])
mt=int(sys.argv[6]); Wt=int(sys.argv[7]); WE=float(sys.argv[8]); seed=int(sys.argv[9]); TARGET0=int(sys.argv[10]) if len(sys.argv)>10 else None
sys.argv=[sys.argv[0],base,'0,0,0,0',out,'40']
src=open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med/ls2b.py').read().split("# singles & pairs")[0]
exec(src)
TARGET=TARGET0 if TARGET0 is not None else val0+1
print('target',TARGET,flush=True)
uE=6.607144099952047e-10; uF=np.array([0.005362359438152787,0.005314435769446963,0.005105837588730707,0.005368888888888889])
DV=np.array([m[2] for m in mv],dtype=np.int64); DE=np.array([m[3] for m in mv],dtype=np.int64)
DD=np.array([m[4] for m in mv],dtype=np.int64); I_=np.array([m[0] for m in mv])
r=DV-uE*DE-DD@uF
keep=np.where(r>=-Rm)[0]
models=sorted(set(I_[keep].tolist())); rnd=random.Random(seed); rnd.shuffle(models)
half={m:(k%2) for k,m in enumerate(models)}
Es=cap-E0; sl=np.array([sl0[t] for t in TY],dtype=np.int64)
print('kept moves',len(keep),'models',len(models),'Es',Es,'sl',sl,flush=True)
def subsets(idx):
    idx=sorted(idx,key=lambda k:-r[k])
    res=[()]
    for k in range(1,K+1):
        for c in itertools.combinations(idx,k):
            if sum(r[j] for j in c)<-Rtot: continue
            if len(set(I_[j] for j in c))<k: continue
            res.append(c)
    return res
t0=time.time()
A=[k for k in keep if half[I_[k]]==0]; B=[k for k in keep if half[I_[k]]==1]
SA=subsets(A); SB=subsets(B)
print('subsets',len(SA),len(SB),round(time.time()-t0),flush=True)
def vecs(S):
    V=np.zeros((len(S),5)); dv=np.zeros(len(S),dtype=np.int64); rr=np.zeros(len(S))
    for n,c in enumerate(S):
        for j in c:
            V[n,0]+=DE[j]; V[n,1:]+=DD[j]; dv[n]+=DV[j]; rr[n]+=r[j]
    return V,dv,rr
VA,dvA,rA=vecs(SA); VB,dvB,rB=vecs(SB)
W=np.array([WE]+[Wt]*4,float)
target=np.array([Es-WE/2]+[sl[k]-mt-Wt/2 for k in range(4)],float)  # sum of moves vector should be near target
tree=cKDTree(VB/W)
found=[]
qs=(target-VA)/W
ordA=np.argsort(-rA)
for n0 in range(0,len(ordA),20000):
    blk=ordA[n0:n0+20000]
    res=tree.query_ball_point(qs[blk],r=0.5,p=np.inf)
    for a,lst in zip(blk,res):
        for b in lst:
            g=dvA[a]+dvB[b]
            if val0+g>=TARGET: found.append((int(g),a,b))
print('matches with gain>=1',len(found),round(time.time()-t0),flush=True)
found.sort(reverse=True)
plans=[]; seenp=set()
for g,a,b in found:
    combo=SA[a]+SB[b]
    key=tuple(sorted((mv[k][0],mv[k][1]) for k in combo))
    if key in seenp: continue
    seenp.add(key)
    tot=np.zeros(5); 
    for k in combo: tot[0]+=DE[k]; tot[1:]+=DD[k]
    if tot[0]>Es: continue
    if val0+g<TARGET: continue
    if not exact_ok(combo): continue
    rr=dict(role)
    for k in combo: rr[mv[k][0]]=mv[k][1]
    p={'full':[i for i in range(nM) if rr[i]=='F'],'half':[i for i in range(nM) if rr[i]=='H'],'source':sorted(srcs),
       'srcreq':[[i,t] for i in tg for t in tg[i]],'obj':int(val0+g),'combo':list(key)}
    plans.append(p)
    if len(plans)>=30: break
print('exact-ok plans',[(p['obj'],p['combo']) for p in plans[:8]],flush=True)
json.dump(plans,open(out,'w')); print('wrote',len(plans),round(time.time()-t0),flush=True)
