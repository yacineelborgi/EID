import sys, json
from ptype import *
plan=json.load(open('../s6/m0_plan.json'))
t=sys.argv[1]; theta=int(sys.argv[2])
bins,items=type_data(plan,t)
small=[b for b in bins if b[1]<=theta]
big=[b for b in bins if b[1]>theta]
totF=sum(s for s,_,fr in items if fr)
LFbig=sum(b[1] for b in big if b[3])
print(len(small),len(items),'free_cap',totF-LFbig, 'tot',sum(s for s,_,_ in items),'Lbig',sum(b[1] for b in big))
r=mip_assign(small,items,tl=float(sys.argv[3]),free_cap=totF-LFbig if len(sys.argv)<5 else None,verbose=True)
print(r if r is None else r[0], r and r[3], r and r[4])
