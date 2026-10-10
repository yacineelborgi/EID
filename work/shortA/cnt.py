import json,math,sys
from ptype import type_data
plan=json.load(open(sys.argv[1]))
for t in 'ntic':
  bins,items=type_data(plan,t)
  mx=max(s for s,_,_ in items)
  for pool in ('F','A'):
    bb=[b for b in bins if (b[3] or pool=='A')]
    it=sorted([s for s,_,f in items if (f or pool=='A')],reverse=True)
    small=[b for b in bb if b[1]<=3000]; big=[b for b in bb if b[1]>3000]
    need=sum(math.ceil(b[1]/mx) for b in big)
    print(t,pool,'items',len(it),'small bins',len(small),'big',len(big),'ceil-need',need,'items left',len(it)-len(small), 'lb hist', [sum(1 for b in big if lo<b[1]<=hi) for lo,hi in ((3000,10000),(10000,20000),(20000,30000),(30000,40000),(40000,200000))])
