"""MIP-LNS on plans with exact small-bin matching (theta) + margins MM. Improving plans (>= TA) saved to lnsS_plans.json
usage: lnsS.py base_sol theta K NF TL IT seed TA"""
import sys, json, time, random, os, subprocess
from collections import defaultdict
import numpy as np, highspy
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/sA')
import mip6
from matchS import add_match
base=sys.argv[1]; theta=int(sys.argv[2]); K=int(sys.argv[3]); NF=int(sys.argv[4]); TL=float(sys.argv[5]); IT=int(sys.argv[6])
rnd=random.Random(int(sys.argv[7])); TA=int(sys.argv[8])
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); cap=inst['energyCap']; M=inst['models']; D=inst['datasets']; nM=len(M)
TH=tuple(int(x) for x in open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med/th_fine.txt').read().strip().split(','))
if os.environ.get('THC'): TH=tuple(int(x) for x in os.environ['THC'].split(','))
mip6.SC=1.0
MM=os.environ.get('MM'); mm={t:int(v) for t,v in zip('ntic',MM.split(','))} if MM else None
mo=mip6.build(inst,exact=True,thetas=TH)
A,c,lb0,ub0,rlo,rhi,xcols,info=add_match(mo,inst,theta=theta,K=K,mm=mm)
ei=mo.names.index('energy')
A=A.tocsr().tolil(); A[ei,:]=A[ei,:]/cap; A=A.tocsc(); rhi=rhi.copy(); rhi[ei]=rhi[ei]/cap
nV=mo.nV; N=A.shape[1]
def mk(lb,ub,integer):
    h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('threads',1)
    lp=highspy.HighsLp(); lp.num_col_=N; lp.num_row_=A.shape[0]
    lp.col_cost_=-c; lp.col_lower_=lb; lp.col_upper_=ub; lp.row_lower_=rlo; lp.row_upper_=rhi
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=A.indptr; lp.a_matrix_.index_=A.indices; lp.a_matrix_.value_=A.data
    if integer: lp.integrality_=[highspy.HighsVarType.kInteger]*nV+[highspy.HighsVarType.kContinuous]*(N-nV)
    h.passModel(lp); return h
h=mk(lb0,ub0,False); h.run(); B=-h.getInfo().objective_function_value; d=-np.array(h.getSolution().col_dual)
print('LP',round(B,3),flush=True)
zcols=defaultdict(list)
for k,(i,t,L,isb) in enumerate(mo.zlist): zcols[i].append(mo.zo+k)
gcols={i:[mo.fo+i,mo.ho+i,mo.so+i]+zcols[i] for i in range(nM)}
FX=float(os.environ.get('FX','12'))
flex=[i for i in range(nM) if min(abs(d[mo.fo+i]),abs(d[mo.ho+i]),abs(d[mo.fo+i]-d[mo.ho+i]),abs(d[mo.so+i]))<=FX]
print('flex',len(flex),flush=True)
def x_of(S):
    srcs={s for s,_ in S['modelMappings']}; tg=defaultdict(set)
    for s,t in S['modelMappings']: tg[t].add(next(iter(M[s]['dataRequirements'])))
    cp=defaultdict(int); used=set()
    for dd,m in S['dataMappings']: used.add(m); cp[m]|=D[dd]['isCopyrighted']
    used|=set(tg); x0=np.zeros(N)
    for i in used-srcs:
        if i in tg or cp[i]: x0[mo.ho+i]=1
        else: x0[mo.fo+i]=1
    for i in srcs: x0[mo.so+i]=1
    for k,(i,t,L,isb) in enumerate(mo.zlist):
        if i in tg and t in tg[i]: x0[mo.zo+k]=1
    lb=lb0.copy(); ub=ub0.copy(); lb[:nV]=x0[:nV]; ub[:nV]=x0[:nV]
    h=mk(lb,ub,False); h.run(); st=h.modelStatusToString(h.getModelStatus())
    xs=np.array(h.getSolution().col_value)
    if len(xs): x0[nV:]=xs[nV:]
    return x0, st
def plan_of(x):
    p=mip6.plan_from_x(mo,x); p['obj']=int(round(c[:nV]@np.round(x[:nV]))); p['small']={}
    for q,(i,t,did,s) in enumerate(xcols):
        if x[mo.nTot+q]>0.5: p['small'].setdefault(t,{})[str(i)]=did
    return p
S=json.load(open(base)); x0,st=x_of(S); best=int(round(c[:nV]@x0[:nV])); print('base',best,st,flush=True)
if st!='Optimal': sys.exit()
found=[]; seen=set()
DF=os.environ.get('DIFF'); dset=set()
if DF:
    q=json.load(open(DF)); q=q[0] if isinstance(q,list) else q
    st={i:('f' if x0[mo.fo+i]>.5 else 'h' if x0[mo.ho+i]>.5 else 's' if x0[mo.so+i]>.5 else '-') for i in range(nM)}
    st2={i:'-' for i in range(nM)}
    for i in q['full']: st2[i]='f'
    for i in q['half']: st2[i]='h'
    for i in q['source']: st2[i]='s'
    dset={i for i in range(nM) if st[i]!=st2[i]}; print('diff models',len(dset),flush=True)
for it in range(IT):
    mode=it%3
    if mode==0: free=set(rnd.sample(flex,min(len(flex),NF)))
    elif mode==1:  # models with free-i / small bins focus
        tt=rnd.choice('ntic'); pool=[i for i in flex if tt in M[i]['dataRequirements']]
        free=set(rnd.sample(pool,min(len(pool),NF)))
    else:  # value-band neighbourhood
        v0=rnd.choice([M[i]['value'] for i in flex]); free=set(sorted(flex,key=lambda i:abs(M[i]['value']-v0)*(0.5+rnd.random()))[:NF])
    free|=dset
    lb=lb0.copy(); ub=ub0.copy()
    cols=np.array([j for i in range(nM) if i not in free for j in gcols[i]],dtype=np.int64)
    lb[cols]=x0[cols]; ub[cols]=x0[cols]
    h=mk(lb,ub,True); h.setOptionValue('time_limit',TL); h.setOptionValue('mip_rel_gap',0.0); h.setOptionValue('mip_abs_gap',0.5); h.setOptionValue('random_seed',it)
    s=highspy.HighsSolution(); s.col_value=list(map(float,x0)); s.value_valid=True; h.setSolution(s)
    h.run(); x=np.array(h.getSolution().col_value)
    obj=int(round(c[:nV]@np.round(x[:nV]))) if len(x) else -1
    print(f'it {it} m{mode} obj {obj} bnd {-h.getInfo().mip_dual_bound:.1f} {h.modelStatusToString(h.getModelStatus())[:8]}',flush=True)
    if obj>=TA:
        p=plan_of(x); key=tuple(sorted(p['full']))+(-1,)+tuple(sorted(p['half']))
        if key not in seen:
            seen.add(key); found.append(p); json.dump(found,open('lnsS_plans.json','w')); print('   saved plan',obj,len(found)-1,flush=True)
