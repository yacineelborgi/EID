import sys, numpy as np, highspy, time
from state import *
S,used,src,tg,F,H=load()
bad={i for i,m in enumerate(M) if any(r['upperBound']<r['lowerBound'] for r in m['dataRequirements'].values())}
fixedH={t for t in tg}  # targets with sources: keep
cand=[i for i in range(len(M)) if i not in F and i not in src and i not in bad and i not in fixedH and M[i]['value']//2>0]
budget=CAP-energy(F,fixedH,src)
print('cand',len(cand),'budget',budget, 'curH val', sum(M[i]['value']//2 for i in H-fixedH))
h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('threads',1)
h.setOptionValue('time_limit',float(sys.argv[1]) if len(sys.argv)>1 else 120.)
h.setOptionValue('mip_rel_gap',0.0)
lp=highspy.HighsLp(); n=len(cand)
lp.num_col_=n; lp.num_row_=1
lp.col_cost_=np.array([M[i]['value']//2 for i in cand],float)
lp.col_lower_=np.zeros(n); lp.col_upper_=np.ones(n)
lp.row_lower_=np.array([-np.inf]); lp.row_upper_=np.array([float(budget)])
lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
lp.a_matrix_.start_=np.arange(n+1); lp.a_matrix_.index_=np.zeros(n,int); lp.a_matrix_.value_=np.array([float(M[i]['energyCost']) for i in cand])
lp.sense_=highspy.ObjSense.kMaximize; lp.integrality_=[highspy.HighsVarType.kInteger]*n
h.passModel(lp)
sol=highspy.HighsSolution(); sol.col_value=[1.0 if i in H else 0.0 for i in cand]; sol.value_valid=True
h.setSolution(sol)
t0=time.time(); h.run()
info=h.getInfo(); x=h.getSolution().col_value
sel=[cand[j] for j in range(n) if x[j]>0.5]
E=sum(M[i]['energyCost'] for i in sel)
print('obj',info.objective_function_value,'bound',info.mip_dual_bound,'E ok',E<=budget, time.time()-t0)
