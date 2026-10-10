import json, exm
from collections import defaultdict
M=exm.M; D=exm.D
S=json.load(open('../out_lb/2_medium_submission.json'))
F,H,Sx=exm.sol_status(S)
sl={'n':77,'t':43,'i':909,'c':110}
used=defaultdict(int)
for d,m in S['dataMappings']:
    if m in F: used[(m,D[d]['dataType'])]+=D[d]['size']
unf=lambda i:[t for t,r in M[i]['dataRequirements'].items() if not (r['upperBound']>=r['lowerBound'] and r['upperBound']>=exm.MINF[t])]
W=int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 40
P=[]
for i in H:
    if unf(i): continue
    need={t:r['lowerBound']+(W if r['lowerBound']<=3000 else 0) for t,r in M[i]['dataRequirements'].items()}
    P.append((M[i]['value']-M[i]['value']//2,i,need))
Q=[(M[i]['value']-M[i]['value']//2,i,{t:used[(i,t)] for t in M[i]['dataRequirements']}) for i in F]
res=[]
for gp,p,need in P:
    # single
    if all(need.get(t,0)<=sl[t] for t in 'ntic'): res.append((gp,p,None))
    for lq,q,fr in Q:
        if gp-lq<1: continue
        if all(need.get(t,0)<=sl[t]+fr.get(t,0) for t in 'ntic'): res.append((gp-lq,p,q))
res.sort(reverse=True); print(len(res)); print(res[:20])
res2=[]
import itertools
for gp,p,need in P:
    dfc={t:need.get(t,0)-sl[t] for t in 'ntic'}
    if max(dfc.values())<=0: continue
    Qs=[q for q in Q if q[0]<gp-1+1 and q[1]!=p]
    for a in range(len(Qs)):
        la,qa,fa=Qs[a]
        for b in range(a+1,len(Qs)):
            lb_,qb,fb=Qs[b]
            if gp-la-lb_<1: continue
            if all(dfc[t]<=fa.get(t,0)+fb.get(t,0) for t in 'ntic'): res2.append((gp-la-lb_,p,qa,qb))
res2.sort(reverse=True); print('triples',len(res2),res2[:10])
# two promotions + one demotion
res3=[]
for x in range(len(P)):
    for y in range(x+1,len(P)):
        g=P[x][0]+P[y][0]; need={t:P[x][2].get(t,0)+P[y][2].get(t,0) for t in 'ntic'}
        dfc={t:need[t]-sl[t] for t in 'ntic'}
        for lq,q,fr in Q:
            if g-lq>=1 and all(dfc[t]<=fr.get(t,0) for t in 'ntic'): res3.append((g-lq,P[x][1],P[y][1],q))
res3.sort(reverse=True); print('PPQ',len(res3),res3[:10])
