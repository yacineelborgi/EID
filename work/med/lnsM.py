"""MIP-LNS for 2_medium: multi-threshold model with free-data margin; fix all but a random subset of models
to the incumbent, solve sub-MIP, pack improving plans (pipeX), adopt if the packed score improves.
usage: lnsM.py base_sol margin nfree tl iters seed"""
import sys, json, time, random, os, subprocess
from collections import defaultdict
import numpy as np, highspy
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
Dd='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med'
base=sys.argv[1]; MG=int(sys.argv[2]); NF=int(sys.argv[3]); TL=float(sys.argv[4]); IT=int(sys.argv[5]); rnd=random.Random(int(sys.argv[6]))
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']; nM=len(M)
mo=mip6.build(inst,marg={('F',t):MG for t in 'ntic'})
def x_of(S):
    srcs={s for s,_ in S['modelMappings']}; tg=defaultdict(set)
    for s,t in S['modelMappings']: tg[t].add(next(iter(M[s]['dataRequirements'])))
    cp=defaultdict(int); used=set()
    for dd,m in S['dataMappings']: used.add(m); cp[m]|=D[dd]['isCopyrighted']
    used|=set(tg); x0=np.zeros(mo.nTot)
    for i in used-srcs:
        if i in tg or cp[i]: x0[mo.ho+i]=1
        else: x0[mo.fo+i]=1
    for i in srcs: x0[mo.so+i]=1
    for k,(i,t,L,isb) in enumerate(mo.zlist):
        if i in tg and t in tg[i]: x0[mo.zo+k]=1
    h=mip6.to_highs(mo,integer=False)
    lb=mo.lb.copy(); ub=mo.ub.copy()
    h.changeColsBounds(mo.nV,np.arange(mo.nV,dtype=np.int32),x0[:mo.nV],x0[:mo.nV]); h.run()
    xs=np.array(h.getSolution().col_value); x0[mo.nV:]=xs[mo.nV:]
    return x0, h.modelStatusToString(h.getModelStatus())
zcols=defaultdict(list)
for k,(i,t,L,isb) in enumerate(mo.zlist): zcols[i].append(mo.zo+k)
sys.path.insert(0,"/home/user/EID")
from test_solution import getSolutionScore
import pickle
C=pickle.load(open(Dd+'/lpm_med.npz.cert.pkl','rb')); rcs=[float(v) for v in C['rcs']]
FX=float(os.environ.get('FX','12'))
flex=[i for i in range(nM) if min(abs(rcs[mo.fo+i]),abs(rcs[mo.ho+i]),abs(rcs[mo.fo+i]-rcs[mo.ho+i]),abs(rcs[mo.so+i]))<=FX]
print('flexible models',len(flex),flush=True)
TXT=open('/home/user/EID/datasets/2_medium.json').read()
S=json.load(open(base)); best,ok,_=getSolutionScore(json.dumps(S),TXT,show_details=False)
x0,st=x_of(S); print('base',best,'warm',st,flush=True)
for it in range(IT):
    free=set(rnd.sample(flex,min(len(flex),int(os.environ.get('NFLEX','100000')))))|set(rnd.sample(range(nM),NF))
    h=mip6.to_highs(mo,integer=True)
    h.setOptionValue('time_limit',TL); h.setOptionValue('mip_rel_gap',0.0); h.setOptionValue('random_seed',it)
    cols=[]
    for i in range(nM):
        if i in free: continue
        cols += [mo.fo+i, mo.ho+i, mo.so+i] + zcols[i]
    cols=np.array(cols,dtype=np.int32)
    h.changeColsBounds(len(cols),cols,x0[cols],x0[cols])
    s=highspy.HighsSolution(); s.col_value=list(map(float,x0)); s.value_valid=True; h.setSolution(s)
    h.run(); info=h.getInfo(); x=np.array(h.getSolution().col_value)
    obj=int(round(mo.cost[:mo.nV]@np.round(x[:mo.nV]))) if len(x) else -1
    print(f'it {it} sub-MIP obj {obj} bound {info.mip_dual_bound:.2f} status {h.modelStatusToString(h.getModelStatus())}',flush=True)
    if obj>best:
        p=mip6.plan_from_x(mo,x); p['obj']=obj
        pf=f'{Dd}/lns_plan.json'; json.dump([p],open(pf,'w'))
        o=subprocess.run(['python3',Dd+'/pipeX.py',pf,'0',Dd+'/best_lns.json','2'],capture_output=True,text=True).stdout
        print('   pack:',[l for l in o.split('\n') if 'score' in l][:2],flush=True)
        if os.path.exists(Dd+'/best_lns.json'):
            S2=json.load(open(Dd+'/best_lns.json')); sc,ok,_=getSolutionScore(json.dumps(S2),TXT,show_details=False)
            if ok and sc>best:
                best=sc; x0,st=x_of(S2); print('   NEW BEST',best,'warm',st,flush=True)
                print('  ',subprocess.run(['python3',Dd+'/saveout.py','2_medium',Dd+'/best_lns.json','272312'],capture_output=True,text=True).stdout.strip()[-90:],flush=True)
