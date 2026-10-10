import sys, json, numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,'.')
from ptype import type_data
def single_cost(small, items, BIGC=10**9):
    C=np.full((len(small),len(items)),BIGC,dtype=np.int64)
    sz=np.array([s for s,_,_ in items]); fr=np.array([f for _,_,f in items])
    for q,(k,L,U,F) in enumerate(small):
        ok=(sz>=L)&(sz<=U)
        if F: ok&=fr
        C[q,ok]=sz[ok]
    r,c=linear_sum_assignment(C)
    return C[r,c].sum()
if __name__=='__main__':
  plan=json.load(open(sys.argv[1]))
  for t in sys.argv[2]:
    bins,items=type_data(plan,t)
    tot=sum(s for s,_,_ in items); totF=sum(s for s,_,f in items if f)
    out=[]
    for theta in [int(x) for x in sys.argv[3].split(',')]:
      small=[b for b in bins if b[1]<=theta]; big=[b for b in bins if b[1]>theta]
      a=tot-sum(b[1] for b in big)-single_cost(small,items)
      sF=[b for b in small if b[3]]; fitems=[x for x in items if x[2]]
      f=totF-sum(b[1] for b in big if b[3])-single_cost(sF,fitems)
      out.append((theta,a,f))
    print(t,'(theta, A-slack, F-slack) single-item:',out,flush=True)
