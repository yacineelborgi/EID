import json
from ptype import *
plan=json.load(open('../s6/m0_plan.json'))
tp=TypePack(plan,'n',0,log=print)
bins,items=tp.bins,tp.items
small=[b for b in range(len(bins)) if bins[b][1]<=3000]
r=match_small([bins[b] for b in small],items)
for q,b in enumerate(small):
    if r[q] is not None: tp.set_bin(b,[r[q]])
pool=SortedList((items[j][0],j) for j in tp.pool() if items[j][2])
print(len(pool), sum(s for s,_ in pool), pool[:20])
pk=Packer([],tp.rng,K=300,G0=20000); pk.pool=pool
big=sorted([b for b in range(len(bins)) if bins[b][1]>3000 and bins[b][3]],key=lambda b:-bins[b][1])
nf=0
for b in big:
    k,L,U,F=bins[b]
    r=pk.fill(L,U,0)
    if r is None:
        nf+=1
        if nf<5: print('fail',L,U,len(pool),sum(s for s,_ in pool), [s for s,_ in pool][:30], [s for s,_ in pool][-10:])
print('nf',nf)
