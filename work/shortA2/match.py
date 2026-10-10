import sys, json, numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,'.')
from ptype import type_data
plan=json.load(open(sys.argv[1]))
for t in sys.argv[2]:
  for theta in [int(x) for x in sys.argv[3].split(',')]:
    bins,items=type_data(plan,t)
    small=[b for b in bins if b[1]<=theta]; big=[b for b in bins if b[1]>theta]
    tot=sum(s for s,_,_ in items)
    BIGC=10**9
    C=np.full((len(small),len(items)),BIGC,dtype=np.int64)
    for q,(k,L,U,F) in enumerate(small):
        for j,(s,_,fr) in enumerate(items):
            if L<=s<=U and (fr or not F): C[q,j]=s
    r,c=linear_sum_assignment(C)
    cost=C[r,c].sum()
    print(t,theta,'room',tot-sum(b[1] for b in big),'single-item min cons',cost if cost<BIGC else 'inf','Lsmall',sum(b[1] for b in small),flush=True)
