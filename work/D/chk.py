"""Necessary check of plans: per type, min consumption of small F-bins (L<=theta) via exact MIP vs cap - sum large L.
usage: chk.py plans.json idxs theta [tl]"""
import sys, json, numpy as np, highspy
from scipy.sparse import coo_matrix
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']
def smallfill(bins, items, tl=60, need=None):
    # bins [(L,U)], items [(s,id)] sorted; min consumption
    R=[];C=[];V=[];lo=[];hi=[];cost=[];nr=0; var=[]
    for b,(L,U) in enumerate(bins):
        cols=[]
        for q,(s,did) in enumerate(items):
            if s>U: break
            var.append((b,q)); cols.append((len(var)-1,s)); cost.append(s)
        for j,s in cols: R.append(nr);C.append(j);V.append(s)
        lo.append(L);hi.append(U);nr+=1
        for j,s in cols: R.append(nr);C.append(j);V.append(1.0 if s>=L else 0.5)
        lo.append(1);hi.append(np.inf);nr+=1
    byq={}
    for j,(b,q) in enumerate(var): byq.setdefault(q,[]).append(j)
    for q,js in byq.items():
        for j in js: R.append(nr);C.append(j);V.append(1)
        lo.append(-np.inf);hi.append(1);nr+=1
    A=coo_matrix((V,(R,C)),shape=(nr,len(var))).tocsc()
    h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('threads',1); h.setOptionValue('time_limit',tl)
    lp=highspy.HighsLp(); lp.num_col_=len(var); lp.num_row_=nr; lp.col_cost_=np.array(cost,float); lp.col_lower_=np.zeros(len(var)); lp.col_upper_=np.ones(len(var))
    lp.row_lower_=np.array(lo,float); lp.row_upper_=np.array(hi,float)
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=A.indptr; lp.a_matrix_.index_=A.indices; lp.a_matrix_.value_=A.data
    lp.integrality_=[highspy.HighsVarType.kInteger]*len(var)
    h.passModel(lp)
    if need is not None: h.setOptionValue('objective_bound', need+0.5)
    h.run(); info=h.getInfo(); st=h.modelStatusToString(h.getModelStatus())
    x=h.getSolution().col_value if info.primal_solution_status==2 else None
    fill=None
    if x is not None:
        fill={}
        for j,(b,q) in enumerate(var):
            if x[j]>0.5: fill.setdefault(b,[]).append(items[q][1])
    return st, (info.objective_function_value if x is not None else None), info.mip_dual_bound, fill
def check(plan, theta=3000, tl=60, types='ntic', verbose=True):
    full=set(plan['full']); res={}
    for t in types:
        items=sorted((d['size'],d['id']) for d in D if d['dataType']==t and not d['isCopyrighted'])
        cap=sum(s for s,_ in items)
        bl=[(M[i]['dataRequirements'][t]['lowerBound'],M[i]['dataRequirements'][t]['upperBound'],i) for i in full if t in M[i]['dataRequirements']]
        small=[(L,U,i) for L,U,i in bl if L<=theta]; large=sum(L for L,U,i in bl if L>theta)
        avail=cap-large
        st,obj,bnd,fill=smallfill([(L,U) for L,U,i in small],items,tl)
        sl=avail-sum(L for L,U,i in small)
        res[t]=(len(small), sl, None if obj is None else obj-sum(L for L,U,i in small), bnd-sum(L for L,U,i in small), avail-(obj if obj is not None else bnd))
        if verbose: print(f' {t}: small {len(small)} slack(sumL) {sl} minwaste {res[t][2]} wasteLB {res[t][3]:.0f} margin {res[t][4]:.0f} {st}',flush=True)
    return res
if __name__=='__main__':
    P=json.load(open(sys.argv[1])); P=[P] if isinstance(P,dict) else P
    idx=range(len(P)) if sys.argv[2]=='all' else [int(x) for x in sys.argv[2].split(',')]
    th=int(sys.argv[3]); tl=float(sys.argv[4]) if len(sys.argv)>4 else 60
    for k in idx:
        print('plan',k,'obj',P[k].get('obj'),flush=True); check(P[k],th,tl)
