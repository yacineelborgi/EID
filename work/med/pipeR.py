"""plan -> pack5 per-type packing -> lnsfix rescue (MIP with donors) -> finalize -> greedy_add. Saves best legit."""
import sys, json, random, os, time
S='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad'
sys.path.insert(0,S+'/e5'); sys.path.insert(0,S+'/s6'); sys.path.insert(0,'/home/user/EID')
os.environ.setdefault('NDONOR','40')
import pack5, lnsfix
from solve import finalize, greedy_add, score_of, strict_ok
INST='/home/user/EID/datasets/2_medium.json'
txt=open(INST).read(); inst=json.loads(txt); M=inst['models']
plans=json.load(open(sys.argv[1]))
if isinstance(plans,dict): plans=[plans]
idx=[int(x) for x in sys.argv[2].split(',')]
out=sys.argv[3]
nseeds=int(sys.argv[4]) if len(sys.argv)>4 else 3
best=-1
if os.path.exists(out):
    best,ok,_=score_of(txt,json.load(open(out)))
for k in idx:
    p=plans[k]; print('plan',k,'obj',p.get('obj'),flush=True)
    for seed in range(nseeds):
        t0=time.time()
        dm,failed,st=pack5.pack(inst,p,theta=1,seed=seed,order='lb_asc')
        nf0=len(failed)
        if failed:
            failed=lnsfix.rescue(inst,p,dm,set(failed),random.Random(seed),n_donor=int(os.environ['NDONOR']),tl=5.0,tbudget=200.0)
        dm2,mm,full,half,us=finalize(inst,p,dm,failed)
        pre=sum(M[i]['value'] for i in full)+sum(M[i]['value']//2 for i in half)
        added=greedy_add(inst,dm2,mm,full,half,us)
        sol={"dataMappings":[[d,m] for d,m in dm2.items()],"modelMappings":mm}
        sc,ok,msg=score_of(txt,sol); ok=ok and strict_ok(inst,sol)
        print(f' seed {seed}: failed {nf0}->{len(failed)} pre {pre} added {added} score {sc} {ok} {msg} {time.time()-t0:.0f}s',flush=True)
        if ok and sc>best:
            best=sc; json.dump(sol,open(out,'w')); print('  wrote',out,sc,flush=True)
