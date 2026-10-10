"""Fast necessary/sufficient-ish check: per type, single-item min-cost assignment of small F-bins (L<=theta), gives feasible fill (upper bound on consumption).
usage: chk1.py plans.json idxs theta"""
import sys, json, numpy as np
from scipy.optimize import linear_sum_assignment
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']
ITEMS={t:sorted((d['size'],d['id']) for d in D if d['dataType']==t and not d['isCopyrighted']) for t in 'ntic'}
def single(plan,t,theta):
    items=ITEMS[t]; cap=sum(s for s,_ in items); full=plan['full']
    bl=[(M[i]['dataRequirements'][t]['lowerBound'],M[i]['dataRequirements'][t]['upperBound'],i) for i in full if t in M[i]['dataRequirements']]
    small=[x for x in bl if x[0]<=theta]; large=sum(x[0] for x in bl if x[0]>theta)
    S=np.array([s for s,_ in items],float)
    Cm=np.full((len(small),len(items)),1e9)
    for b,(L,U,i) in enumerate(small):
        ok=(S>=L)&(S<=U); Cm[b,ok]=S[ok]
    r,c=linear_sum_assignment(Cm); cons=Cm[r,c].sum()
    sl=cap-large-sum(x[0] for x in small)
    return len(small), sl, cons-sum(x[0] for x in small), cap-large-cons, {small[b][2]:[items[q][1]] for b,q in zip(r,c)}
if __name__=='__main__':
    P=json.load(open(sys.argv[1])); P=[P] if isinstance(P,dict) else P
    idx=range(len(P)) if sys.argv[2]=='all' else [int(x) for x in sys.argv[2].split(',')]
    th=int(sys.argv[3])
    for k in idx:
        out=[]
        for t in 'ntic':
            n,sl,w,mg,_=single(P[k],t,th); out.append(f'{t}:{n} sl {sl} w {w:.0f} mg {mg:.0f}')
        print('plan',k,P[k].get('obj'),' | '.join(out),flush=True)
