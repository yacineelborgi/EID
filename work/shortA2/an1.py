import json,sys
sys.path.insert(0,'/home/user/EID')
from ptype import type_data, M, D, INST
plan=json.load(open(sys.argv[1]))
for t in 'ntic':
    bins,items=type_data(plan,t)
    tot=sum(s for s,_,_ in items); totF=sum(s for s,_,f in items if f)
    sL=sum(b[1] for b in bins); sLF=sum(b[1] for b in bins if b[3])
    small=[b for b in bins if b[1]<=3000]
    # greedy single item min waste for small bins (ignore F)
    import bisect
    sz=sorted(s for s,_,_ in items)
    used=[False]*len(sz); w=0
    for b in sorted(small,key=lambda b:-b[1]):
        pass
    print(t,'nb',len(bins),'nF',sum(b[3] for b in bins),'small',len(small),'tot',tot,'sumL',sL,'slack',tot-sL,'Ffree',totF,'sLF',sLF,'Fslack',totF-sLF,
          'nitems',len(items),'smallL',sum(b[1] for b in small))
