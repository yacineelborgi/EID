"""LNS on exact small-bin model (all types). usage: lnsy.py sol.json T_fix theta W tmax seed out_plans.json"""
import sys, json, pickle, time, os, random
import numpy as np, highspy
SP='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/'
sys.path.insert(0,SP+'s6'); sys.path.insert(0,SP+'D')
import mip6
from sol2plan import sol2plan
from scipy.sparse import vstack, hstack, csr_matrix, coo_matrix
solf=sys.argv[1]; T=int(sys.argv[2]); THE=int(sys.argv[3]); W=int(sys.argv[4]); tmax=float(sys.argv[5]); seed=int(sys.argv[6]); outp=sys.argv[7]
rng=random.Random(seed)
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); cap=inst['energyCap']; M=inst['models']; D=inst['datasets']
if os.environ.get('PLANIDX'):
    P0=json.load(open(solf)); P0=[P0] if isinstance(P0,dict) else P0; plan0=P0[int(os.environ['PLANIDX'])]; fills0={}
else:
    S0=json.load(open(solf)); plan0,fills0,by0=sol2plan(S0)
TH=tuple(int(x) for x in open(SP+'med/th_fine.txt').read().strip().split(','))
C=pickle.load(open(SP+'med/lpm_fine.npz.cert.pkl','rb')); B=C['B']; rcs=C['rcs']; ye=C['ye']
mip6.SC=1.0
mo=mip6.build(inst,exact=True,thetas=TH); nV=mo.nV; nM=mo.nM
bud=float(B-T)+1e-7
lb=mo.lb.copy(); ub=mo.ub.copy()
r=np.array([float(x) for x in rcs])
x0=np.zeros(mo.nTot)
for i in plan0['full']: x0[mo.fo+i]=1
for i in plan0['half']: x0[mo.ho+i]=1
for i in plan0['source']: x0[mo.so+i]=1
zk={(i,t):k for k,(i,t,L,isb) in enumerate(mo.zlist)}
for i,t in plan0['srcreq']: x0[mo.zo+zk[(i,t)]]=1
nfix=0; bad=0
for j in range(nV):
    if r[j]>bud: lb[j]=ub[j]; nfix+=1; bad+=x0[j]!=ub[j]
    elif r[j]<-bud: ub[j]=lb[j]; nfix+=1; bad+=x0[j]!=lb[j]
print('fixed',nfix,'incumbent conflicts',bad,flush=True)
if bad:
    for j in range(nV):
        if x0[j]<lb[j] or x0[j]>ub[j]: lb[j]=mo.lb[j]; ub[j]=mo.ub[j]
A=mo.A.tocsr(); rlo=mo.rlo.copy(); rhi=mo.rhi.copy()
if not os.environ.get('NOIMP'):
  for k in range(len(ye)):
    v=float(ye[k])
    if v>0 and np.isfinite(mo.rhi[k]): rlo[k]=max(rlo[k], mo.rhi[k]-bud/v-1e-6)
    elif v<0 and np.isfinite(mo.rlo[k]): rhi[k]=min(rhi[k], mo.rlo[k]+bud/(-v)+1e-6)
const=sum(r[j]*mo.ub[j] if r[j]>0 else r[j]*mo.lb[j] for j in range(nV))
brow=np.zeros(mo.nTot); brow[:nV]=-r[:nV]
A2=vstack([A,csr_matrix(brow.reshape(1,-1))]).tocsr()
rlo=np.append(rlo,-np.inf); rhi=np.append(rhi,bud-const+1e-6)
ei=mo.names.index('energy')
A2=A2.tolil(); A2[ei,:]=A2[ei,:]/cap; A2=A2.tocsr()
rlo[ei]=rlo[ei]/cap; rhi[ei]=rhi[ei]/cap
nC0=mo.nTot
yinfo=[]; R=[];Cc=[];V=[];RL=[];RH=[]; nr=[0]
def row(ent,lo,hi):
    for j,v in ent: R.append(nr[0]); Cc.append(j); V.append(v)
    RL.append(lo); RH.append(hi); nr[0]+=1
y0=[]
EXTRA=int(os.environ.get('EXTRA','0'))
for t in 'ntic':
    items=sorted((d['size'],d['id']) for d in D if d['dataType']==t and not d['isCopyrighted'])
    capF=sum(s for s,_ in items)
    bins=[]; large=[]
    for k,(i,tt,L,isb) in enumerate(mo.zlist):
        if tt!=t or isb: continue
        xf=mo.fo+i
        if ub[xf]<=0: continue
        U=M[i]['dataRequirements'][t]['upperBound']
        (bins if L<=THE else large).append((i,L,U,xf))
    marg=sum(sum(D[d]['size'] for d in fills0.get(t,{}).get(i,[]))-L for (i,L,U,xf) in large if x0[xf]>0.5)
    if os.environ.get('MARGS'): marg=int(os.environ['MARGS'].split(',')['ntic'.index(t)])
    itemcols={}
    for (i,L,U,xf) in bins:
        inc=set(fills0.get(t,{}).get(i,[])) if x0[xf]>0.5 else set()
        cols=[]
        for s,did in items:
            if s>U: break
            if s>L+W and did not in inc: continue
            j=nC0+len(yinfo); yinfo.append((t,i,did,s)); cols.append((j,s)); itemcols.setdefault(did,[]).append(j)
            y0.append(1.0 if did in inc else 0.0)
            row([(j,1),(xf,-1)],-np.inf,0)
        row([(j,s) for j,s in cols]+[(xf,-L)],0,np.inf)
        row([(j,s) for j,s in cols]+[(xf,-U)],-np.inf,0)
        row([(j,1.0 if s>=L else 0.5) for j,s in cols]+[(xf,-1)],0,np.inf)
    for did,js in itemcols.items(): row([(j,1) for j in js],-np.inf,1)
    row([(xf,L) for (i,L,U,xf) in large]+[(nC0+q,yinfo[q][3]) for q in range(len(yinfo)) if yinfo[q][0]==t],-np.inf,capF-marg-EXTRA)
    print(t,'small',len(bins),'large',len(large),'margin',marg,'ny',len(yinfo),flush=True)
nY=len(yinfo)
A2=hstack([A2,csr_matrix((A2.shape[0],nY))]).tocsr()
A3=vstack([A2,coo_matrix((V,(R,Cc)),shape=(nr[0],nC0+nY)).tocsr()]).tocsc()
rlo=np.concatenate([rlo,RL]); rhi=np.concatenate([rhi,RH])
cost=np.concatenate([-mo.cost,np.zeros(nY)]); LB=np.concatenate([lb,np.zeros(nY)]); UB=np.concatenate([ub,np.ones(nY)])
NC=nC0+nY
h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('threads',1); h.setOptionValue('random_seed',seed)
lp=highspy.HighsLp(); lp.num_col_=NC; lp.num_row_=A3.shape[0]
lp.col_cost_=cost; lp.col_lower_=LB; lp.col_upper_=UB; lp.row_lower_=rlo; lp.row_upper_=rhi
lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=A3.indptr; lp.a_matrix_.index_=A3.indices; lp.a_matrix_.value_=A3.data
lp.integrality_=[highspy.HighsVarType.kInteger]*nV+[highspy.HighsVarType.kContinuous]*(nC0-nV)+[highspy.HighsVarType.kInteger]*nY
h.passModel(lp)
intcols=np.concatenate([np.arange(nV),np.arange(nC0,NC)]).astype(np.int32)
y0=np.array(y0)
def extract(x):
    p=mip6.plan_from_x(mo,x); p['obj']=int(round(mo.cost[:nV]@np.round(x[:nV])))
    sm={}
    for q in range(nY):
        if x[nC0+q]>0.5: t,i,did,s=yinfo[q]; sm.setdefault(t,{}).setdefault(str(i),[]).append(did)
    p['small']=sm; return p
def solve_nb(freeM, tl, cutoff, start=True):
    lo=LB.copy(); hi=UB.copy()
    fx=np.ones(nM,bool); fx[list(freeM)]=False
    for off in (mo.fo,mo.ho,mo.so):
        idx=off+np.where(fx)[0]; lo[idx]=np.clip(x0[idx],LB[idx],UB[idx]); hi[idx]=lo[idx]
    for k,(i,t,L,isb) in enumerate(mo.zlist):
        if fx[i]: lo[mo.zo+k]=hi[mo.zo+k]=np.clip(x0[mo.zo+k],LB[mo.zo+k],UB[mo.zo+k])
    h.changeColsBounds(NC,np.arange(NC,dtype=np.int32),lo,hi)
    h.setOptionValue('time_limit',tl); h.setOptionValue('objective_bound',-(cutoff-0.5))
    if start:
        vals=np.concatenate([x0[:nV],y0]); h.setSolution(len(intcols),intcols,vals)
    h.run()
    st=h.modelStatusToString(h.getModelStatus()); info=h.getInfo()
    if info.primal_solution_status==2:
        x=np.array(h.getSolution().col_value); return st,-info.objective_function_value,x
    return st,None,None
if not os.environ.get('PLANIDX'):
    st,ob,x=solve_nb([],60,plan0['obj'])
    print('incumbent check',st,ob,flush=True)
inc=int(os.environ.get('INC',plan0['obj'])); plans=[]; t0=time.time(); it=0
cand=[i for i in range(nM) if any(UB[o+i]>LB[o+i] for o in (mo.fo,mo.ho,mo.so))]
print('candidates',len(cand),flush=True)
inm=set(plan0['full'])|set(plan0['half'])|set(plan0['source'])
outc=[i for i in cand if i not in inm]; inc_c=[i for i in cand if i in inm]
Ls={tt:[M[i]['dataRequirements'][tt]['lowerBound'] for i in inc_c if tt in M[i]['dataRequirements']] for tt in 'ntic'}
diffM=[]
if os.environ.get('NBMODE')=='3':
    h2=highspy.Highs(); h2.setOptionValue('output_flag',False); h2.setOptionValue('threads',1)
    lp.integrality_=[]; h2.passModel(lp); h2.run(); xl=np.array(h2.getSolution().col_value)
    print('LP',-h2.getInfo().objective_function_value,flush=True)
    diffM=[i for i in cand if any(abs(xl[o+i]-x0[o+i])>0.05 for o in (mo.fo,mo.ho,mo.so))]
    for i in diffM: print(' m',i,'v',M[i]['value'],'E%.4f'%(M[i]['energyCost']/cap),'req',{t:(q['lowerBound'],q['upperBound']) for t,q in M[i]['dataRequirements'].items()},'lp f/h/s %.2f %.2f %.2f'%(xl[mo.fo+i],xl[mo.ho+i],xl[mo.so+i]),'inc',x0[mo.fo+i],x0[mo.ho+i],x0[mo.so+i],flush=True)
    np.save(outp+'.xl.npy',xl)
    if os.environ.get('LPONLY'): sys.exit()
    del h2
print('diffM',len(diffM),flush=True)
print('out cands',len(outc),'in cands',len(inc_c),flush=True)
KS=[int(v) for v in os.environ.get('KS','60,100,150,200').split(',')]
if os.environ.get('GLOBAL'):
    h.setOptionValue('mip_heuristic_effort',float(os.environ.get('HEFF','0.3')))
    h.setOptionValue('output_flag',True); h.setOptionValue('log_to_console',False); h.setOptionValue('log_file',outp+'.highs.log')
    sols=[]
    def cb(e):
        x=np.array(e.data_out.mip_solution); p=extract(x); sols.append(p); json.dump(sols,open(outp,'w')); print('improving',p['obj'],round(time.time()-t0),flush=True)
    h.cbMipImprovingSolution.subscribe(cb)
    st,ob,x=solve_nb(range(nM),tmax,int(os.environ.get('GLOBAL')),start=True)
    print('GLOBAL END',st,ob,'dual',-h.getInfo().mip_dual_bound,flush=True); sys.exit()
while time.time()-t0<tmax:
    it+=1
    k=rng.choice(KS)
    if os.environ.get('NBMODE')=='1':
        nb=set(outc)|set(rng.sample(inc_c,min(k,len(inc_c))))
    elif os.environ.get('NBMODE')=='3':
        nb=set(diffM)|set(rng.sample(cand,min(k,len(cand))))
    elif os.environ.get('NBMODE')=='2':
        tt=rng.choice('ntic'); Lc=rng.choice(Ls[tt]); 
        near=sorted(cand,key=lambda i: abs(M[i]['dataRequirements'][tt]['lowerBound']-Lc) if tt in M[i]['dataRequirements'] else 1e9)
        nb=set(near[:k])|set(rng.sample(outc,min(len(outc),k//2)))
    else:
        nb=set(rng.sample(cand,min(k,len(cand))))
    st,ob,x=solve_nb(nb,rng.choice([30,60]),inc+1,start=False)
    if ob is not None and ob>=inc+1-1e-6:
        p=extract(x); plans.append(p); json.dump(plans,open(outp,'w'))
        print(f'it {it} k {k} FOUND {p["obj"]} {time.time()-t0:.0f}s',flush=True)
        if os.environ.get('MOVE'):
            x0[:nV]=np.round(x[:nV]); y0[:]=np.round(x[nC0:]); inc=p['obj']
    elif it%5==0: print(f'it {it} {st} {time.time()-t0:.0f}s',flush=True)
print('END',inc,len(plans),flush=True)
