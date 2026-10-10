"""pipeM + MIP rescue inside per-type packing: failed full bins are re-packed jointly with the highest-waste
(and random) donor bins of the same type via an assignment MIP (lnsfix.solve_assign), before demotion."""
import sys, json, os, time, random
os.chdir('/home/user/EID'); sys.path.insert(0,'/home/user/EID')
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import med_solve, exact_pack
from lnsfix import solve_assign
from solve import score_of, strict_ok
_orig=exact_pack.pack_type
NDON=int(os.environ.get('NDON','30')); TRIES=int(os.environ.get('TRIES','8')); TL=float(os.environ.get('TL','8'))
def pack_type_x(bins, items, seed=0, order='lb_desc', **kw):
    fill, still, pool = _orig(bins, items, seed=seed, order=order, **kw)
    rnd=random.Random(seed)
    binfo={k:(lb,ub) for k,lb,ub in bins}
    still2=[]
    for key in still:
        lb,ub=binfo[key]
        if ub<lb: still2.append(key); continue
        ok=False
        for tr in range(TRIES):
            keys=list(fill.keys())
            waste=sorted(keys,key=lambda k:-(sum(x[0] for x in fill[k][2])-fill[k][0]))
            nd=NDON+5*tr
            don=waste[:nd//2]+rnd.sample(waste[nd//2:],min(nd-nd//2,len(waste)-nd//2))
            its=list(pool)+[x for d in don for x in fill[d][2]]
            bb=[(lb,ub)]+[(fill[d][0],fill[d][1]) for d in don]
            res=solve_assign(bb,[x[0] for x in its],lambda b,j:True,tl=TL)
            if res is None: continue
            newp=set(its)
            fill[key]=(lb,ub,[its[j] for j in res[0]]); newp.difference_update(fill[key][2])
            for b,d in enumerate(don):
                fill[d]=(fill[d][0],fill[d][1],[its[j] for j in res[b+1]]); newp.difference_update(fill[d][2])
            pool=newp; ok=True
            print(f'    rescued bin {key} lb {lb} try {tr}',flush=True)
            break
        if not ok: still2.append(key)
    return fill, still2, pool
med_solve.pack_type=pack_type_x
INST='/home/user/EID/datasets/2_medium.json'; txt=open(INST).read(); inst=json.loads(txt)
plans=json.load(open(sys.argv[1])); plans=[plans] if isinstance(plans,dict) else plans
idx=[int(x) for x in sys.argv[2].split(',')] if sys.argv[2]!='all' else range(len(plans))
out=sys.argv[3]; seeds=int(sys.argv[4]) if len(sys.argv)>4 else 1
best=-1
if os.path.exists(out): best,ok,_=score_of(txt,json.load(open(out)))
for k in idx:
    p=plans[k]; print('plan',k,'obj',p.get('obj'),flush=True)
    for o in ['lb_asc']:
        for seed in range(seeds):
            t0=time.time()
            sol,failed,pre,added=med_solve.build(inst,p,order=o,seed=seed,verbose=False)
            sc,ok,msg=score_of(txt,sol); ok=ok and strict_ok(inst,sol)
            print(f'  {o} seed {seed}: failed {len(failed)} pre {pre} added {added} score {sc} {ok} {msg} {time.time()-t0:.0f}s',flush=True)
            if ok and sc>best:
                best=sc; json.dump(sol,open(out,'w')); print('  wrote',out,sc,flush=True)
