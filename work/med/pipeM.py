"""plans file -> med_solve.build (exact packing, demotion, greedy add) with several orders/seeds; keeps best legit."""
import sys, json, os, time
os.chdir('/home/user/EID'); sys.path.insert(0,'/home/user/EID')
from med_solve import build
from solve import score_of, strict_ok
INST='/home/user/EID/datasets/2_medium.json'; txt=open(INST).read(); inst=json.loads(txt)
plans=json.load(open(sys.argv[1])); plans=[plans] if isinstance(plans,dict) else plans
idx=[int(x) for x in sys.argv[2].split(',')] if sys.argv[2]!='all' else range(len(plans))
out=sys.argv[3]; seeds=int(sys.argv[4]) if len(sys.argv)>4 else 2
best=-1
if os.path.exists(out): best,ok,_=score_of(txt,json.load(open(out)))
for k in idx:
    p=plans[k]; print('plan',k,'obj',p.get('obj'),flush=True)
    for o in ['lb_asc','lb_desc']:
        for seed in range(seeds):
            t0=time.time()
            sol,failed,pre,added=build(inst,p,order=o,seed=seed,verbose=False)
            sc,ok,msg=score_of(txt,sol); ok=ok and strict_ok(inst,sol)
            print(f'  {o} seed {seed}: failed {len(failed)} pre {pre} added {added} score {sc} {ok} {msg} {time.time()-t0:.0f}s',flush=True)
            if ok and sc>best:
                best=sc; json.dump(sol,open(out,'w')); print('  wrote',out,sc,flush=True)
