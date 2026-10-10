"""tgt.py + exact small-bin fill vars (valid relaxation). usage: tgtm.py T tl out.json types theta seed [lponly]
For each type t in types, F-bins (full models) with L<=theta: y[b,d] for free items d (s_d<=U_b):
 sum s y in [L x, U x]; sum_{s>=L} y + 1/2 sum_{s<L} y >= x; y<=x; sum_b y[b,d]<=1; cap: sum_large L x + sum s y <= capF."""
import sys, json, pickle, time, os
import numpy as np, highspy
SP='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/'
sys.path.insert(0,SP+'s6')
import mip6
from scipy.sparse import vstack, hstack, csr_matrix, coo_matrix
T=int(sys.argv[1]); tl=float(sys.argv[2]); out=sys.argv[3]; TYP=sys.argv[4]; THE=[int(x) for x in sys.argv[5].split(',')]
seed=int(sys.argv[6]); LPONLY=len(sys.argv)>7 and sys.argv[7]=='lp'
if len(THE)==1: THE=THE*len(TYP)
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); cap=inst['energyCap']; M=inst['models']; D=inst['datasets']
TH=tuple(int(x) for x in open(SP+'med/th_fine.txt').read().strip().split(','))
C=pickle.load(open(SP+'med/lpm_fine.npz.cert.pkl','rb'))
B=C['B']; rcs=C['rcs']; ye=C['ye']
mip6.SC=1.0
mo=mip6.build(inst,exact=True,thetas=TH)
assert mo.nTot==len(rcs)
bud=float(B-T)+1e-7
lb=mo.lb.copy(); ub=mo.ub.copy(); nV=mo.nV
r=np.array([float(x) for x in rcs])
nfix=0
for j in range(nV):
    if r[j]>bud: lb[j]=ub[j]; nfix+=1
    elif r[j]<-bud: ub[j]=lb[j]; nfix+=1
A=mo.A.tocsr(); rlo=mo.rlo.copy(); rhi=mo.rhi.copy()
nimp=0
for k in range(len(ye)):
    v=float(ye[k])
    if v>0 and np.isfinite(mo.rhi[k]): rlo[k]=max(rlo[k], mo.rhi[k]-bud/v-1e-6); nimp+=1
    elif v<0 and np.isfinite(mo.rlo[k]): rhi[k]=min(rhi[k], mo.rlo[k]+bud/(-v)+1e-6); nimp+=1
const=sum(r[j]*mo.ub[j] if r[j]>0 else r[j]*mo.lb[j] for j in range(nV))
brow=np.zeros(mo.nTot); brow[:nV]=-r[:nV]
A2=vstack([A,csr_matrix(brow.reshape(1,-1))]).tocsr()
rlo=np.append(rlo,-np.inf); rhi=np.append(rhi,bud-const+1e-6)
ei=mo.names.index('energy')
A2=A2.tolil(); A2[ei,:]=A2[ei,:]/cap; A2=A2.tocsr()
rlo[ei]=rlo[ei]/cap if np.isfinite(rlo[ei]) else rlo[ei]; rhi[ei]=rhi[ei]/cap
nR0=A2.shape[0]; nC0=mo.nTot
# ---- small-bin fill vars
yinfo=[]  # (t, model i, item id, size)
R=[];Cc=[];V=[];RL=[];RH=[]
nr=[0]
def row(ent,lo,hi):
    for j,v in ent: R.append(nr[0]); Cc.append(j); V.append(v)
    RL.append(lo); RH.append(hi); nr[0]+=1
for t,the in zip(TYP,THE):
    items=sorted((d['size'],d['id']) for d in D if d['dataType']==t and not d['isCopyrighted'])
    capF=sum(s for s,_ in items)
    bins=[]; large=[]
    for k,(i,tt,L,isb) in enumerate(mo.zlist):
        if tt!=t or isb: continue
        xf=mo.fo+i
        if ub[xf]<=0: continue
        U=M[i]['dataRequirements'][t]['upperBound']
        (bins if L<=the else large).append((i,L,U,xf))
    itemcols={}
    for (i,L,U,xf) in bins:
        cols=[]
        for s,did in items:
            if s>U: break
            j=nC0+len(yinfo); yinfo.append((t,i,did,s)); cols.append((j,s)); itemcols.setdefault(did,[]).append(j)
            row([(j,1),(xf,-1)],-np.inf,0)
        row([(j,s) for j,s in cols]+[(xf,-L)],0,np.inf)
        row([(j,s) for j,s in cols]+[(xf,-U)],-np.inf,0)
        row([(j,1.0 if s>=L else 0.5) for j,s in cols]+[(xf,-1)],0,np.inf)
    for did,js in itemcols.items(): row([(j,1) for j in js],-np.inf,1)
    row([(xf,L) for (i,L,U,xf) in large]+[(nC0+q,yinfo[q][3]) for q in range(len(yinfo)) if yinfo[q][0]==t],-np.inf,capF)
    print(t,'theta',the,'small bins',len(bins),'large',len(large),'y so far',len(yinfo),flush=True)
nY=len(yinfo)
A2=hstack([A2,csr_matrix((A2.shape[0],nY))]).tocsr()
Anew=coo_matrix((V,(R,Cc)),shape=(nr[0],nC0+nY)).tocsr()
A3=vstack([A2,Anew]).tocsc()
rlo=np.concatenate([rlo,RL]); rhi=np.concatenate([rhi,RH])
cost=np.concatenate([-mo.cost,np.zeros(nY)]); lb=np.concatenate([lb,np.zeros(nY)]); ub=np.concatenate([ub,np.ones(nY)])
print('fixed',nfix,'of',nV,'rows',A3.shape,'nY',nY,flush=True)
h=highspy.Highs(); h.setOptionValue('output_flag',True); h.setOptionValue('log_to_console',False)
h.setOptionValue('log_file',out+'.highs.log'); h.setOptionValue('threads',1)
lp=highspy.HighsLp(); lp.num_col_=A3.shape[1]; lp.num_row_=A3.shape[0]
lp.col_cost_=cost; lp.col_lower_=lb; lp.col_upper_=ub; lp.row_lower_=rlo; lp.row_upper_=rhi
lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=A3.indptr; lp.a_matrix_.index_=A3.indices; lp.a_matrix_.value_=A3.data
if not LPONLY:
    lp.integrality_=[highspy.HighsVarType.kInteger]*nV+[highspy.HighsVarType.kContinuous]*(nC0-nV)+[highspy.HighsVarType.kInteger]*nY
h.passModel(lp)
t0=time.time()
if LPONLY:
    h.run(); print('LP',h.modelStatusToString(h.getModelStatus()),-h.getInfo().objective_function_value,round(time.time()-t0),flush=True); sys.exit()
h.setOptionValue('time_limit',tl); h.setOptionValue('mip_rel_gap',0.0); h.setOptionValue('mip_abs_gap',0.5)
h.setOptionValue('random_seed',seed); h.setOptionValue('objective_bound',-(T-0.5))
h.setOptionValue('mip_heuristic_effort',float(os.environ.get('HEFF','0.1')))
sols=[]
E=np.array([m['energyCost'] for m in M])
def cb(e):
    x=np.array(e.data_out.mip_solution)
    p=mip6.plan_from_x(mo,x); p['obj']=int(round(mo.cost[:nV]@np.round(x[:nV])))
    en=int(E[p['full']].sum()+E[p['half']].sum()+E[p['source']].sum()); p['energy_ok']=en<=cap
    sm={}
    for q in range(nY):
        if x[nC0+q]>0.5: t,i,did,s=yinfo[q]; sm.setdefault(t,{}).setdefault(str(i),[]).append(did)
    p['small']=sm
    p['t']=time.time()-t0; sols.append(p); print('improving',p['obj'],'energy ok',en<=cap,'at',round(p['t']),flush=True)
    json.dump(sols,open(out,'w'))
h.cbMipImprovingSolution.subscribe(cb)
h.run(); info=h.getInfo()
print('FINAL',h.modelStatusToString(h.getModelStatus()),'obj',-info.objective_function_value if sols else None,'dual bound',-info.mip_dual_bound,'time',round(time.time()-t0),flush=True)
