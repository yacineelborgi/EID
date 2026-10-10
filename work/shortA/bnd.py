import sys, json
from ptype import *
plan=json.load(open(sys.argv[1]))
for t in sys.argv[2]:
  for theta in [int(x) for x in sys.argv[3].split(',')]:
    bins,items=type_data(plan,t)
    small=[b for b in bins if b[1]<=theta]
    big=[b for b in bins if b[1]>theta]
    tot=sum(s for s,_,_ in items); totF=sum(s for s,_,fr in items if fr)
    LFbig=sum(b[1] for b in big if b[3]); Lbig=sum(b[1] for b in big)
    r=mip_assign(small,items,tl=float(sys.argv[4]),free_cap=totF-LFbig)
    # rerun to get bound even if no sol
    import ptype; print(t,theta,'room',tot-Lbig,'res',ptype.LAST_BOUND if r is None else (r[3],r[4]),flush=True)
