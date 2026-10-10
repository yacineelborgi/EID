import sys, json
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/e5')
sys.path.insert(0,'/home/user/EID')
import pack5
INST='/home/user/EID/datasets/2_medium.json'
txt=open(INST).read(); inst=json.loads(txt)
plans=json.load(open(sys.argv[1]))
if isinstance(plans,dict): plans=[plans]
idx=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [len(plans)-1]
out=sys.argv[3] if len(sys.argv)>3 else 'best_med.json'
best=-1
for k in idx:
    p=plans[k]; print('plan',k,'obj',p.get('obj'),flush=True)
    for th in [1,300]:
        sc,sol=pack5.evaluate(inst,txt,p,th,seeds=(0,1),verbose=True)
        if sol and sc>best:
            best=sc; json.dump(sol,open(out,'w')); print('  wrote',out,sc,flush=True)
