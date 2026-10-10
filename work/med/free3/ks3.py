"""Rigorous upper bound for 3_free (no copyrighted data): relax ALL data constraints.
Each model -> one 0-1 item:
  good model (every req ub>=lb): full option, value v, weight e.
  bad model (some req ub<lb): only half is possible, every bad req must be sourced by a distinct mono
     model of that type with ub>=lb -> value v//2, weight e + sum_{bad t} minsrc_t  (relaxation: cheapest source).
Real solutions map to relaxation solutions with >= value and <= energy (sources dropped, voluntarily-sourced
good models mapped to full). Then knapsack LP (exact Fractions) + reduced-cost exhaustive search for target T."""
import json, sys
from fractions import Fraction as Fr
I=json.load(open('/home/user/EID/datasets/3_free.json')); M=I['models']; cap=I['energyCap']
assert all(d['isCopyrighted']==0 for d in I['datasets'])
minsrc={}
for t in 'ntic':
    c=[m['energyCost'] for m in M if len(m['dataRequirements'])==1 and t in m['dataRequirements'] and m['dataRequirements'][t]['upperBound']>=m['dataRequirements'][t]['lowerBound']]
    minsrc[t]=min(c) if c else None
print('minsrc',minsrc)
items=[]
for i,m in enumerate(M):
    rq=m['dataRequirements']; badt=[t for t,r in rq.items() if r['upperBound']<r['lowerBound']]
    if not badt: items.append((m['value'],m['energyCost'],i,'full'))
    else:
        if any(minsrc[t] is None for t in badt): continue
        items.append((m['value']//2, m['energyCost']+sum(minsrc[t] for t in badt), i, 'half'))
# items with value 0 never help; items with weight 0 and value>0 always in
items=[it for it in items if it[0]>0]
zero=[it for it in items if it[1]==0]
pos=[it for it in items if it[1]>0]
pos.sort(key=lambda it: Fr(it[0],it[1]), reverse=True)
V0=sum(it[0] for it in zero); W=0; V=V0
for k,it in enumerate(pos):
    if W+it[1]<=cap: W+=it[1]; V+=it[0]
    else: crit=k; break
u=Fr(pos[crit][0],pos[crit][1])
L=V+u*(cap-W)
print('items',len(items),'zero-weight',len(zero),'LP bound',float(L),'u',float(u),'crit',pos[crit])
T=int(sys.argv[1]) if len(sys.argv)>1 else 523759
budget=L-T
print('target',T,'budget',float(budget))
inS=set(range(crit))
dev=[]
for k,it in enumerate(pos):
    d=(Fr(it[0])-u*it[1]) if k<crit else (u*it[1]-it[0])
    if k==crit: d=Fr(0)
    if d<=budget: dev.append((d,k))
print('candidate deviations',len(dev))
# exhaustive: base = items 0..crit-1 in; deviations flip membership; crit item free (dev 0)
import itertools
best=None; cnt=0
cand=sorted(dev)
def rec(idx, dsum, flips):
    global best, cnt
    if idx==len(cand):
        cnt+=1
        Wx=W; Vx=V
        for k in flips:
            it=pos[k]
            if k<crit: Wx-=it[1]; Vx-=it[0]
            else: Wx+=it[1]; Vx+=it[0]
        if Wx<=cap and Vx>=T:
            if best is None or Vx>best[0]: best=(Vx,list(flips))
        return
    d,k=cand[idx]
    rec(idx+1,dsum,flips)
    if dsum+d<=budget:
        rec(idx+1,dsum+d,flips+[k])
rec(0,Fr(0),[])
print('enumerated',cnt,'best >= target:',best)
