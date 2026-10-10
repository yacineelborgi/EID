exec(open('enumpq.py').read().split('res=[]')[0])
# source swaps: current source s (type t) -> half/full ; mono model m (type t) in F/H -> source
SW=[]
for s in Sx:
    t=list(M[s]['dataRequirements'])[0]
    gs=M[s]['value']//2  # s -> half
    gsf=M[s]['value'] if not unf(s) else None
    for m in F|H:
        if list(M[m]['dataRequirements'])!=[t] or unf(m): continue
        lm=M[m]['value'] if m in F else M[m]['value']//2
        fr={t:used[(m,t)]} if m in F else {}
        SW.append((gs-lm,s,m,'h',fr,{}))
        if gsf is not None:
            r=M[s]['dataRequirements'][t]; SW.append((gsf-lm,s,m,'f',fr,{t:r['lowerBound']+(W if r['lowerBound']<=3000 else 0)}))
out=[]
for g0,s,m,k,fr,nd in SW:
    for gp,p,need in P+[(0,None,{})]:
        if p==s: continue
        tot={t:need.get(t,0)+nd.get(t,0) for t in 'ntic'}
        if g0+gp>=1 and all(tot[t]<=sl[t]+fr.get(t,0) for t in 'ntic'): out.append((g0+gp,s,m,k,p))
        for lq,q,fq in Q:
            if q==m or g0+gp-lq<1: continue
            if all(tot[t]<=sl[t]+fr.get(t,0)+fq.get(t,0) for t in 'ntic'): out.append((g0+gp-lq,s,m,k,p,q))
out.sort(key=lambda x:-x[0]); print(len(out), out[:15])
