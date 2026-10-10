"""MIP of relaxation R (margin 0, tiny cuts). Saves every improving integer plan.
usage: python mipR.py time_limit tau tag [margin_n,t,i,c]"""
import sys, json, pickle, time
sys.path.insert(0,'.')
from rel import *
from collections import defaultdict
tl=float(sys.argv[1]); tau=int(sys.argv[2]); tag=sys.argv[3]
margin = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv)>4 else [0,0,0,0]
inst = load('/home/user/EID/datasets/2_medium.json'); M=inst['models']; D=inst['datasets']
R = Rel(inst, tau={t:tau for t in TYPES}); R.build()
for j,t in enumerate(TYPES): R.rhi[R.rowid[('F',t)]] = R.F[t]-margin[j]
# warm start from current best solution
S=json.load(open('/home/user/EID/solutions/2_medium_submission.json'))
fo,ho,so,zo,co=R.off
srcs={s for s,_ in S['modelMappings']}
tg=defaultdict(set)
for s,t in S['modelMappings']: tg[t].add(next(iter(M[s]['dataRequirements'])))
cp=defaultdict(int); used=set()
for dd,m in S['dataMappings']: used.add(m); cp[m]|=D[dd]['isCopyrighted']
used|=set(tg)
x0=np.zeros(R.nV)
for i in used-srcs:
    if i in tg or cp[i]: x0[ho+i]=1
    else: x0[fo+i]=1
for i in srcs: x0[so+i]=1
for k,(i,t,lb,isb) in enumerate(R.zlist):
    if i in tg and t in tg[i]: x0[zo+k]=1
for j,t in enumerate(TYPES):
    kk=int(sum(x0[fo+i] for i in R.tiny[t])); x0[co+j]=max(int(R.P[t][kk]), sum(x0[fo+i]*M[i]['dataRequirements'][t]['lowerBound'] for i in R.tiny[t]))
print('start obj', R.c@x0, flush=True)
sols=[]
def plan_of(x):
    xi=np.round(x).astype(int)
    p={"full":[],"half":[],"source":[],"srcreq":[]}
    for i in range(R.nM):
        if xi[fo+i]: p['full'].append(i)
        if xi[ho+i]: p['half'].append(i)
        if xi[so+i]: p['source'].append(i)
    for k,(i,t,lb,isb) in enumerate(R.zlist):
        if xi[zo+k]: p['srcreq'].append([i,t])
    p['obj']=int(round(R.c[:co]@xi[:co]))
    return p
def cb(e):
    x=np.array(e.data_out.mip_solution)
    p=plan_of(x); p['t']=time.time()-t0
    sols.append(p)
    print('improving', p['obj'], 'at', round(p['t']), flush=True)
    json.dump(sols, open(f'plansR_{tag}.json','w'))
t0=time.time()
h=highspy.Highs()
# replicate R.solve but with callback
h.setOptionValue("output_flag", True); h.setOptionValue("time_limit", tl); h.setOptionValue("threads",1)
h.setOptionValue("mip_rel_gap",0.0); h.setOptionValue("mip_abs_gap",0.0)
h.setOptionValue("log_file", f"mipR_{tag}.highs.log")
lp=highspy.HighsLp(); A=R.A
lp.num_col_=R.nV; lp.num_row_=A.shape[0]; lp.sense_=highspy.ObjSense.kMaximize
lp.col_cost_=R.c; lp.col_lower_=R.lbv; lp.col_upper_=R.ubv; lp.row_lower_=R.rlo; lp.row_upper_=R.rhi
lp.a_matrix_.format_=highspy.MatrixFormat.kColwise; lp.a_matrix_.start_=A.indptr; lp.a_matrix_.index_=A.indices; lp.a_matrix_.value_=A.data
lp.a_matrix_.num_col_=R.nV; lp.a_matrix_.num_row_=A.shape[0]
integ=[highspy.HighsVarType.kInteger]*R.nV
for j in range(4): integ[co+j]=highspy.HighsVarType.kContinuous
lp.integrality_=integ
h.passModel(lp)
h.cbMipImprovingSolution.subscribe(cb)
s=highspy.HighsSolution(); s.col_value=list(map(float,x0)); s.value_valid=True; h.setSolution(s)
h.run()
info=h.getInfo()
print('FINAL status',h.modelStatusToString(h.getModelStatus()),'obj',info.objective_function_value,'bound',info.mip_dual_bound,'gap',info.mip_gap, flush=True)
