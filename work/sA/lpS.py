"""LP of fine multi-threshold model + exact small-bin matching. usage: lpS.py theta K out.npz"""
import sys, json, time, pickle
import numpy as np, highspy
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/sA')
import mip6
from matchS import add_match
theta=int(sys.argv[1]); K=int(sys.argv[2]); out=sys.argv[3]
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); cap=inst['energyCap']
TH=tuple(int(x) for x in open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med/th_fine.txt').read().strip().split(','))
mip6.SC=1.0
t0=time.time()
mo=mip6.build(inst,exact=True,thetas=TH)
A2,c2,lb2,ub2,rlo2,rhi2,xcols,info=add_match(mo,inst,theta=theta,K=K)
print(info,len(xcols),A2.shape,time.time()-t0,flush=True)
ei=mo.names.index('energy')
A2=A2.tocsr().tolil(); A2[ei,:]=A2[ei,:]/cap; A2=A2.tocsc(); rhi2=rhi2.copy(); rhi2[ei]=rhi2[ei]/cap
h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('threads',1)
lp=highspy.HighsLp(); lp.num_col_=A2.shape[1]; lp.num_row_=A2.shape[0]
lp.col_cost_=c2; lp.col_lower_=lb2; lp.col_upper_=ub2; lp.row_lower_=rlo2; lp.row_upper_=rhi2
lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=A2.indptr; lp.a_matrix_.index_=A2.indices; lp.a_matrix_.value_=A2.data
lp.sense_=highspy.ObjSense.kMaximize
h.passModel(lp); h.run()
print('LP',h.modelStatusToString(h.getModelStatus()),h.getInfo().objective_function_value,time.time()-t0,flush=True)
sol=h.getSolution()
np.savez(out,y=np.array(sol.row_dual),x=np.array(sol.col_value),rc=np.array(sol.col_dual))
