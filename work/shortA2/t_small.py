import json, sys
from ptype import type_data, match_small
plan=json.load(open(sys.argv[1]))
for t in sys.argv[2]:
  bins,items=type_data(plan,t)
  totF=sum(s for s,_,f in items if f); tot=sum(s for s,_,_ in items)
  big=[b for b in bins if b[1]>3000]
  LFb=sum(b[1] for b in big if b[3]); Lb=sum(b[1] for b in big)
  small=[b for b in bins if b[1]<=3000]
  for lam in (0.3,50,300,1000,3000):
    r=match_small(small,items,lam=lam)
    usedF=sum(items[j][0] for j in r if j is not None and items[j][2]); used=sum(items[j][0] for j in r if j is not None)
    print(t,lam,'F slack',totF-LFb-usedF,'A slack',tot-Lb-used,'unmatched',sum(1 for j in r if j is None))
