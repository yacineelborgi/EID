"""Targeted plan MIP: fine multi-threshold model + exact small-bin matching (F pool, single item per small bin).
LP rc-fixing + dual-implied row bounds + budget row for value >= T; MIP with cutoff; saves improving plans.
usage: tgtS.py T tl out.json theta K [seed] [warm.json]"""
import sys, json, time, os
import numpy as np, highspy
from collections import defaultdict
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/sA')
import mip6
from matchS import add_match
from scipy.sparse import vstack, csr_matrix
T=int(sys.argv[1]); tl=float(sys.argv[2]); out=sys.argv[3]; theta=int(sys.argv[4]); K=int(sys.argv[5])
seed=int(sys.argv[6]) if len(sys.argv)>6 else 0
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); cap=inst['energyCap']; M=inst['models']; D=inst['datasets']
TH=tuple(int(x) for x in open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med/th_fine.txt').read().strip().split(','))
mip6.SC=1.0
MG=os.environ.get('MGT')
marg={('F',t):int(v) for t,v in zip('ntic',MG.split(','))} if MG else None
mo=mip6.build(inst,exact=True,thetas=TH,marg=marg)
MM=os.environ.get('MM'); mm={t:int(v) for t,v in zip('ntic',MM.split(','))} if MM else None
A,c,lb0,ub0,rlo,rhi,xcols,info=add_match(mo,inst,theta=theta,K=K,mm=mm)
ei=mo.names.index('energy')
A=A.tocsr().tolil(); A[ei,:]=A[ei,:]/cap; A=A.tocsc(); rhi=rhi.copy(); rhi[ei]=rhi[ei]/cap
nV=mo.nV; N=A.shape[1]
def mk(lb,ub,rl,rh,AA,integer):
    h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('threads',1)
    lp=highspy.HighsLp(); lp.num_col_=AA.shape[1]; lp.num_row_=AA.shape[0]
    lp.col_cost_=-c; lp.col_lower_=lb; lp.col_upper_=ub; lp.row_lower_=rl; lp.row_upper_=rh
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=AA.indptr; lp.a_matrix_.index_=AA.indices; lp.a_matrix_.value_=AA.data
    if integer: lp.integrality_=[highspy.HighsVarType.kInteger]*nV+[highspy.HighsVarType.kContinuous]*(N-nV)
    h.passModel(lp); return h
t0=time.time()
h=mk(lb0,ub0,rlo,rhi,A,False); h.run()
B=-h.getInfo().objective_function_value; sol=h.getSolution()
d=-np.array(sol.col_dual); y=-np.array(sol.row_dual)   # for max c: d = c - A^T y
print('LP',B,round(time.time()-t0),flush=True)
bud=B-T+0.01
lb=lb0.copy(); ub=ub0.copy(); nfix=0
for j in range(nV):
    if d[j]>bud: lb[j]=ub[j]; nfix+=1
    elif d[j]<-bud: ub[j]=lb[j]; nfix+=1
rl=rlo.copy(); rh=rhi.copy(); nimp=0
for k in range(len(y)):
    v=y[k]
    if v>1e-9 and np.isfinite(rhi[k]): rl[k]=max(rl[k], rhi[k]-bud/v-1e-6); nimp+=1
    elif v<-1e-9 and np.isfinite(rlo[k]): rh[k]=min(rh[k], rlo[k]+bud/(-v)+1e-6); nimp+=1
const=sum(d[j]*ub0[j] if d[j]>0 else d[j]*lb0[j] for j in range(nV))
brow=np.zeros(N); brow[:nV]=-d[:nV]
A2=vstack([A.tocsr(),csr_matrix(brow.reshape(1,-1))]).tocsc()
rl=np.append(rl,-np.inf); rh=np.append(rh,bud-const+1e-6)
print('fixed',nfix,'of',nV,'implied',nimp,'bud',round(bud,3),'free models',int(sum(1 for i in range(mo.nM) if ub[mo.fo+i]+ub[mo.ho+i]+ub[mo.so+i]>lb[mo.fo+i]+lb[mo.ho+i]+lb[mo.so+i])),flush=True)

W=os.environ.get('WARM')
if W:
    p=json.load(open(W)); p=p[0] if isinstance(p,list) else p
    x0=np.zeros(N)
    for i in p['full']: x0[mo.fo+i]=1
    for i in p['half']: x0[mo.ho+i]=1
    for i in p['source']: x0[mo.so+i]=1
    sr={(a,b) for a,b in p['srcreq']}
    for k,(i,t,L,isb) in enumerate(mo.zlist):
        if (i,t) in sr: x0[mo.zo+k]=1
    l2=lb.copy(); u2=ub.copy(); bad=np.where((x0[:nV]<lb[:nV]-1e-9)|(x0[:nV]>ub[:nV]+1e-9))[0]; print('warm viol fixings',len(bad),flush=True)
    l2[:nV]=x0[:nV]; u2[:nV]=x0[:nV]
    hw=mk(l2,u2,rl,rh,A2,False); hw.run(); print('warm LP',hw.modelStatusToString(hw.getModelStatus()),-hw.getInfo().objective_function_value,flush=True)
    xs=np.array(hw.getSolution().col_value)
    if len(xs): x0[nV:]=xs[nV:]
h=mk(lb,ub,rl,rh,A2,True)
h.setOptionValue('output_flag',True); h.setOptionValue('log_to_console',False); h.setOptionValue('log_file',out+'.highs.log')
h.setOptionValue('time_limit',tl); h.setOptionValue('mip_rel_gap',0.0); h.setOptionValue('mip_abs_gap',0.5)
h.setOptionValue('random_seed',seed); h.setOptionValue('objective_bound',-(T-0.5))
h.setOptionValue('mip_heuristic_effort',float(os.environ.get('HEFF','0.1')))
sols=[]; t0=time.time()
E=np.array([m['energyCost'] for m in M])
def cb(e):
    x=np.array(e.data_out.mip_solution)
    p=mip6.plan_from_x(mo,x); p['obj']=int(round(c[:nV]@np.round(x[:nV])))
    p['small']={}
    for q,(i,t,did,s) in enumerate(xcols):
        if x[mo.nTot+q]>0.5: p['small'].setdefault(t,{})[str(i)]=did
    en=int(E[p['full']].sum()+E[p['half']].sum()+E[p['source']].sum()); p['energy_ok']=en<=cap
    p['t']=time.time()-t0; sols.append(p); print('improving',p['obj'],'energy ok',en<=cap,'at',round(p['t']),flush=True)
    json.dump(sols,open(out,'w'))
h.cbMipImprovingSolution.subscribe(cb)
if W:
    ss=highspy.HighsSolution(); ss.col_value=list(map(float,x0)); ss.value_valid=True; h.setSolution(ss)
h.run(); info=h.getInfo()
print('FINAL',h.modelStatusToString(h.getModelStatus()),'obj',-info.objective_function_value if sols else None,'dual bound',-info.mip_dual_bound,'time',round(time.time()-t0),flush=True)
