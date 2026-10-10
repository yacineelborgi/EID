"""Rigorous bound for 3_free (no copyrighted data) by dropping every data constraint.
Relaxation R3: per model one role in {off, full (all reqs ub>=lb), source (mono, ub>=lb), half (needs >=1 source;
only useful for 'bad' models, i.e. some req with ub<lb, every bad req must be sourced)}.
Rows: energy <= cap; per type t: #sourced bad reqs of type t <= #sources of type t.
(A good model that is half in a real solution maps to full: higher value, lower energy since its sources go off.)
Lagrangian L(u) computed in exact rationals, then reduced-cost exhaustive search for value >= T."""
import json, sys
from fractions import Fraction as Fr
import numpy as np, highspy
from scipy.sparse import coo_matrix
I=json.load(open('/home/user/EID/datasets/3_free.json')); M=I['models']; cap=I['energyCap']; nM=len(M)
assert all(d['isCopyrighted']==0 for d in I['datasets'])
TY='ntic'
good=[all(r['upperBound']>=r['lowerBound'] for r in m['dataRequirements'].values()) for m in M]
mono=[len(m['dataRequirements'])==1 for m in M]
badt=[[t for t,r in m['dataRequirements'].items() if r['upperBound']<r['lowerBound']] for m in M]
canS=[good[i] and mono[i] for i in range(nM)]
stype=[next(iter(M[i]['dataRequirements'])) if canS[i] else None for i in range(nM)]
# options per model: (name, value, energy, src_vec dict t->coef) ; src row: sum_needs - sum_sources <= 0
def options(i):
    m=M[i]; v=m['value']; e=m['energyCost']; ops=[('off',0,0,{})]
    if good[i]: ops.append(('full',v,e,{}))
    else: ops.append(('half',v//2,e,{t:1 for t in badt[i]}))
    if canS[i]: ops.append(('source',0,e,{stype[i]:-1}))
    return ops
OPS=[options(i) for i in range(nM)]
# LP: variables per (i, option!=off)
var=[]; 
for i in range(nM):
    for k,o in enumerate(OPS[i]):
        if k>0: var.append((i,k))
nv=len(var); rows=[];cols=[];vals=[]
for j,(i,k) in enumerate(var):
    nm,v,e,sv=OPS[i][k]
    rows.append(0); cols.append(j); vals.append(e/1e12)
    for t,c in sv.items(): rows.append(1+TY.index(t)); cols.append(j); vals.append(c)
    rows.append(5+i); cols.append(j); vals.append(1)
A=coo_matrix((vals,(rows,cols)),shape=(5+nM,nv)).tocsc()
h=highspy.Highs(); h.setOptionValue('output_flag',False)
lp=highspy.HighsLp(); lp.num_col_=nv; lp.num_row_=5+nM; lp.sense_=highspy.ObjSense.kMaximize
lp.col_cost_=np.array([OPS[i][k][1] for i,k in var],float); lp.col_lower_=np.zeros(nv); lp.col_upper_=np.ones(nv)
lp.row_lower_=np.full(5+nM,-highspy.kHighsInf); lp.row_upper_=np.array([cap/1e12,0,0,0,0]+[1]*nM,float)
lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=A.indptr; lp.a_matrix_.index_=A.indices; lp.a_matrix_.value_=A.data
h.passModel(lp); h.run()
print('LP',h.getInfo().objective_function_value)
y=h.getSolution().row_dual
uE=Fr(abs(y[0])).limit_denominator(10**15)/10**12
uT={t:Fr(abs(y[1+TY.index(t)])).limit_denominator(10**9) for t in TY}
print('uE',float(uE),'uT',{t:float(x) for t,x in uT.items()})
def prof(o):
    nm,v,e,sv=o
    return Fr(v)-uE*e-sum(uT[t]*c for t,c in sv.items())
L=uE*cap; best=[]; devs=[]
for i in range(nM):
    ps=[prof(o) for o in OPS[i]]; b=max(ps); kb=ps.index(b); L+=b; best.append(kb)
    for k,p in enumerate(ps):
        if k!=kb: devs.append((b-p,i,k))
print('rigorous L(u) =',float(L))
T=int(sys.argv[1]) if len(sys.argv)>1 else 523759
budget=L-T
devs.sort()
cand=[d for d in devs if d[0]<=budget]
print('target',T,'budget',float(budget),'candidate deviations',len(cand), 'ties at 0:', sum(1 for d in devs if d[0]==0))
# exhaustive search: base = argmax options; apply deviations (at most one per model); check energy & source rows
baseE=sum(OPS[i][best[i]][2] for i in range(nM)); baseV=sum(OPS[i][best[i]][1] for i in range(nM))
baseS={t:sum(OPS[i][best[i]][3].get(t,0) for i in range(nM)) for t in TY}
print('base: value',baseV,'energy',baseE,'cap',cap,'src rows',baseS)
cnt=0; hits=[]
def rec(idx,dsum,chosen,used):
    global cnt
    # evaluate
    if idx==len(cand):
        cnt+=1
        E=baseE; V=baseV; S=dict(baseS)
        for (d,i,k) in chosen:
            o0=OPS[i][best[i]]; o1=OPS[i][k]
            E+=o1[2]-o0[2]; V+=o1[1]-o0[1]
            for t in TY: S[t]+=o1[3].get(t,0)-o0[3].get(t,0)
        if E<=cap and all(S[t]<=0 for t in TY) and V>=T: hits.append((V,[(i,k) for d,i,k in chosen]))
        return
    d,i,k=cand[idx]
    rec(idx+1,dsum,chosen,used)
    if i not in used and dsum+d<=budget:
        rec(idx+1,dsum+d,chosen+[(d,i,k)],used|{i})
sys.setrecursionlimit(100000)
if len(cand)<=int(sys.argv[2] if len(sys.argv)>2 else 60): rec(0,Fr(0),[],frozenset()); print('enumerated',cnt,'hits',len(hits),sorted(hits,reverse=True)[:5])
else: print('too many candidates for naive enumeration')
