"""Multi-threshold MIP (mip6 model) on 2_medium, warm-started from the current best; saves improving plans.
usage: mipM.py time_limit tag [thetas] [jmax]"""
import sys, json, time
from collections import defaultdict
import numpy as np, highspy
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
tl=float(sys.argv[1]); tag=sys.argv[2]
TH=tuple(sorted(set(int(x) for x in sys.argv[3].split(',')))) if len(sys.argv)>3 and sys.argv[3] else mip6.THETAS
jmax=int(sys.argv[4]) if len(sys.argv)>4 else 450
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']
mo=mip6.build(inst,thetas=TH,jmax=jmax)
S=json.load(open(sys.argv[5] if len(sys.argv)>5 else '/home/user/EID/solutions/2_medium_submission.json'))
srcs={s for s,_ in S['modelMappings']}; tg=defaultdict(set)
for s,t in S['modelMappings']: tg[t].add(next(iter(M[s]['dataRequirements'])))
cp=defaultdict(int); used=set()
for dd,m in S['dataMappings']: used.add(m); cp[m]|=D[dd]['isCopyrighted']
used|=set(tg)
x0=np.zeros(mo.nTot)
for i in used-srcs:
    if i in tg or cp[i]: x0[mo.ho+i]=1
    else: x0[mo.fo+i]=1
for i in srcs: x0[mo.so+i]=1
for k,(i,t,L,isb) in enumerate(mo.zlist):
    if i in tg and t in tg[i]: x0[mo.zo+k]=1
# complete continuous vars by LP with integers fixed
h=mip6.to_highs(mo,integer=False)
lb=mo.lb.copy(); ub=mo.ub.copy(); lb[:mo.nV]=x0[:mo.nV]; ub[:mo.nV]=x0[:mo.nV]
h.changeColsBounds(mo.nV,np.arange(mo.nV,dtype=np.int32),lb[:mo.nV],ub[:mo.nV])
h.run(); st=h.modelStatusToString(h.getModelStatus())
xs=np.array(h.getSolution().col_value); print('warm LP',st,h.getInfo().objective_function_value,flush=True)
x0[mo.nV:]=xs[mo.nV:]
sols=[]; t0=time.time()
def cb(e):
    x=np.array(e.data_out.mip_solution)
    p=mip6.plan_from_x(mo,x); p['obj']=int(round(mo.cost[:mo.nV]@np.round(x[:mo.nV]))); p['t']=time.time()-t0
    sols.append(p); print('improving',p['obj'],'at',round(p['t']),flush=True)
    json.dump(sols,open(f'plansM_{tag}.json','w'))
h=mip6.to_highs(mo,integer=True)
h.setOptionValue('output_flag',True); h.setOptionValue('log_file',f'mipM_{tag}.highs.log')
h.setOptionValue('time_limit',tl); h.setOptionValue('random_seed',int(__import__('os').environ.get('SEED','0'))); h.setOptionValue('mip_heuristic_effort',float(__import__('os').environ.get('HEFF','0.05'))); h.setOptionValue('mip_rel_gap',0.0); h.setOptionValue('mip_abs_gap',0.0)
h.cbMipImprovingSolution.subscribe(cb)
s=highspy.HighsSolution(); s.col_value=list(map(float,x0)); s.value_valid=True; h.setSolution(s)
h.run(); info=h.getInfo()
print('FINAL',h.modelStatusToString(h.getModelStatus()),'obj',info.objective_function_value,'bound',info.mip_dual_bound,flush=True)
