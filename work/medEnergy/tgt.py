"""Targeted plan search on 2_medium: exact cert (fine thetas) -> reduced-cost fixing for value >= T,
budget row + dual-implied row bounds; MIP (min -value) with cutoff; saves all improving plans.
usage: tgt.py T tl out.json [margF] [seed]"""
import sys, json, pickle, time, os
import numpy as np, highspy
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
T=int(sys.argv[1]); tl=float(sys.argv[2]); out=sys.argv[3]
MG=int(sys.argv[4]) if len(sys.argv)>4 else 0; seed=int(sys.argv[5]) if len(sys.argv)>5 else 0
MGT=os.environ.get('MGT')  # per-type margins n,t,i,c
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); cap=inst['energyCap']
TH=tuple(int(x) for x in open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med/th_fine.txt').read().strip().split(','))
C=pickle.load(open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med/lpm_fine.npz.cert.pkl','rb'))
B=C['B']; rcs=C['rcs']; ye=C['ye']
mip6.SC=1.0
marg={('F',t):MG for t in 'ntic'}
if MGT: marg={('F',t):int(v) for t,v in zip('ntic',MGT.split(','))}
mo=mip6.build(inst,exact=True,thetas=TH,marg=marg)
mo0=mip6.build(inst,exact=True,thetas=TH) if marg and any(marg.values()) else mo
assert mo.nTot==len(rcs)
bud=float(B-T)+1e-7
lb=mo.lb.copy(); ub=mo.ub.copy(); nV=mo.nV
r=np.array([float(x) for x in rcs])
nfix=0
for j in range(nV):
    if r[j]>bud: lb[j]=ub[j]; nfix+=1
    elif r[j]<-bud: ub[j]=lb[j]; nfix+=1
A=mo.A.tocsr(); rlo=mo.rlo.copy(); rhi=mo.rhi.copy()
# dual-implied row bounds (rows of the unmargined model; margins only tighten hi)
nimp=0
for k in range(len(ye)):
    v=float(ye[k])
    if v>0 and np.isfinite(mo0.rhi[k]): rlo[k]=max(rlo[k], mo0.rhi[k]-bud/v-1e-6); nimp+=1
    elif v<0 and np.isfinite(mo0.rlo[k]): rhi[k]=min(rhi[k], mo0.rlo[k]+bud/(-v)+1e-6); nimp+=1
# budget row over integer columns
const=sum(r[j]*ub[j] if r[j]>0 else r[j]*lb[j] for j in range(nV))  # penalty uses original bounds
const=sum(r[j]*mo.ub[j] if r[j]>0 else r[j]*mo.lb[j] for j in range(nV))
brow=np.zeros(mo.nTot); brow[:nV]=-r[:nV]
from scipy.sparse import vstack, csr_matrix
A2=vstack([A,csr_matrix(brow.reshape(1,-1))]).tocsr()
rlo=np.append(rlo,-np.inf); rhi=np.append(rhi,bud-const+1e-6)
# scale energy row
ei=mo.names.index('energy')
A2=A2.tolil(); A2[ei,:]=A2[ei,:]/cap; A2=A2.tocsc()
rlo[ei]=rlo[ei]/cap if np.isfinite(rlo[ei]) else rlo[ei]; rhi[ei]=rhi[ei]/cap
print('fixed',nfix,'of',nV,'implied rows',nimp,'budget',bud,flush=True)
h=highspy.Highs(); h.setOptionValue('output_flag',True); h.setOptionValue('log_to_console',False)
h.setOptionValue('log_file',out+'.highs.log'); h.setOptionValue('threads',1)
lp=highspy.HighsLp(); lp.num_col_=mo.nTot; lp.num_row_=A2.shape[0]
lp.col_cost_=-mo.cost; lp.col_lower_=lb; lp.col_upper_=ub; lp.row_lower_=rlo; lp.row_upper_=rhi
lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=A2.indptr; lp.a_matrix_.index_=A2.indices; lp.a_matrix_.value_=A2.data
lp.integrality_=[highspy.HighsVarType.kInteger]*nV+[highspy.HighsVarType.kContinuous]*(mo.nTot-nV)
h.passModel(lp)
h.setOptionValue('time_limit',tl); h.setOptionValue('mip_rel_gap',0.0); h.setOptionValue('mip_abs_gap',0.5)
h.setOptionValue('random_seed',seed); h.setOptionValue('objective_bound',-(T-0.5))
h.setOptionValue('mip_heuristic_effort',float(os.environ.get('HEFF','0.1')))
sols=[]; t0=time.time()
E=np.array([m['energyCost'] for m in inst['models']])
def cb(e):
    x=np.array(e.data_out.mip_solution)
    p=mip6.plan_from_x(mo,x); p['obj']=int(round(mo.cost[:nV]@np.round(x[:nV])))
    en=int(E[p['full']].sum()+E[p['half']].sum()+E[p['source']].sum()); p['energy_ok']=en<=cap
    p['t']=time.time()-t0; sols.append(p); print('improving',p['obj'],'energy ok',en<=cap,'at',round(p['t']),flush=True)
    json.dump(sols,open(out,'w'))
h.cbMipImprovingSolution.subscribe(cb)
h.run(); info=h.getInfo()
print('FINAL',h.modelStatusToString(h.getModelStatus()),'obj',-info.objective_function_value if sols else None,'dual bound',-info.mip_dual_bound,'time',round(time.time()-t0),flush=True)
