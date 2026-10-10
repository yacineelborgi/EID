import json, sys
import numpy as np
from ptype import type_data, match_small
LAMS=(30,100,200,300,450,600,800,1000,3000)
def joint(plan, t, theta=3000, lams=LAMS):
    bins,items=type_data(plan,t)
    totF=sum(s for s,_,f in items if f); tot=sum(s for s,_,_ in items)
    big=[b for b in bins if b[1]>theta]
    LFb=sum(b[1] for b in big if b[3]); Lb=sum(b[1] for b in big)
    small=[b for b in bins if b[1]<=theta]
    out=[]
    for lam in lams:
        r=match_small(small,items,lam=lam)
        if any(j is None for j in r): out.append((lam,-10**9,-10**9)); continue
        usedF=sum(items[j][0] for j in r if items[j][2]); used=sum(items[j][0] for j in r)
        out.append((lam,int(totF-LFb-usedF),int(tot-Lb-used)))
    best=max(out,key=lambda x:min(x[1],x[2]))
    return best,out
if __name__=='__main__':
    for f in sys.argv[1:]:
        plan=json.load(open(f))
        print(f, plan.get('obj'))
        for t in 'ntic':
            best,out=joint(plan,t)
            print('  ',t,'best',best,'curve',[(l,a,b) for l,a,b in out])
