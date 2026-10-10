import json,sys
sys.path.insert(0,'/home/user/EID')
from ptype import type_data, match_small
plan=json.load(open(sys.argv[1]))
th=int(sys.argv[2]) if len(sys.argv)>2 else 3000
for t in 'ntic':
    bins,items=type_data(plan,t)
    small=[b for b in bins if b[1]<=th]; big=[b for b in bins if b[1]>th]
    r=match_small(small,items)
    nf=sum(x is None for x in r)
    cons=sum(items[j][0] for j in r if j is not None)
    consF=sum(items[j][0] for j in r if j is not None and items[j][2])
    sw=cons-sum(b[1] for b in small)
    tot=sum(s for s,_,_ in items); totF=sum(s for s,_,f in items if f)
    bigL=sum(b[1] for b in big); bigLF=sum(b[1] for b in big if b[3])
    print(t,'small',len(small),'fail',nf,'smallwaste',sw,'big slack',tot-cons-bigL,'bigF slack',totF-consF-bigLF, 'nbig',len(big),'items left',len(items)-len(small))
