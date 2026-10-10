"""Triple-move combo search (see ls2.py) with reduced-value prefilter r = dv - uE dE - sum u_t dd_t >= -R."""
import sys, json, time
import numpy as np
exec(open('ls2.py').read().split("print('moves'")[0])
R=float(sys.argv[5]) if len(sys.argv)>5 else 26
uE=6.607144099952047e-10; uF=np.array([0.005362359438152787,0.005314435769446963,0.005105837588730707,0.005368888888888889])
I_=np.array([m[0] for m in mv]); DV=np.array([m[2] for m in mv],dtype=np.int64); DE=np.array([m[3] for m in mv],dtype=np.int64)
DD=np.array([m[4] for m in mv],dtype=np.int64)
r=DV-uE*DE-DD@uF
keep=np.where(r>=-R)[0]
print('moves',len(mv),'kept',len(keep),'r max',r.max(),flush=True)
I2,DV2,DE2,DD2,r2=I_[keep],DV[keep],DE[keep],DD[keep],r[keep]
Es=cap-E0; sl=np.array([sl0[t] for t in TY],dtype=np.int64)
n=len(keep); found=[]; t0=time.time()
for a in range(n):
    b=np.arange(a+1,n)
    # pair partial
    pr=r2[a]+r2[b]; b=b[pr>=-R]
    if len(b)==0: continue
    for bb in b:
        c=np.arange(bb+1,n)
        dv=DV2[a]+DV2[bb]+DV2[c]
        m=dv>0
        if not m.any(): continue
        c=c[m]
        dE=DE2[a]+DE2[bb]+DE2[c]; m=dE<=Es
        if not m.any(): continue
        c=c[m]
        dd=DD2[a]+DD2[bb]+DD2[c]
        m=np.all(dd<=sl[None,:],axis=1)&(I2[c]!=I2[a])&(I2[c]!=I2[bb])&(I2[a]!=I2[bb])
        for cc in c[m]:
            found.append((int(DV2[a]+DV2[bb]+DV2[cc]),(int(keep[a]),int(keep[bb]),int(keep[cc]))))
    if a%200==0: print(' a',a,'found',len(found),round(time.time()-t0),flush=True)
print('triples found',len(found),time.time()-t0,flush=True)
found.sort(reverse=True)
res=[]
for g,combo in found[:5000]:
    if exact_ok(combo): res.append((g,combo))
    if len(res)>=40: break
print('exact-ok',[(g,[(mv[k][0],mv[k][1]) for k in c]) for g,c in res[:10]],flush=True)
plans=[]
for g,combo in res:
    rr=dict(role)
    for k in combo: rr[mv[k][0]]=mv[k][1]
    p={'full':[i for i in range(nM) if rr[i]=='F'],'half':[i for i in range(nM) if rr[i]=='H'],'source':sorted(srcs),
       'srcreq':[[i,t] for i in tg for t in tg[i]],'obj':val0+g}
    plans.append(p)
json.dump(plans,open(out,'w')); print('wrote',len(plans),'plans',flush=True)
