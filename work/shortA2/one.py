import sys, json, random
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA')
import pipe6a
plan=json.load(open(sys.argv[1])); seed=int(sys.argv[3])
rnd=random.Random(seed)
sol,nf,added=pipe6a.build6(plan,rnd,0.0 if seed==0 else 0.02*(1+seed%5),'lb_asc')
print('failed',nf,'added',added,'score',pipe6a.evaluate(sol),flush=True)
json.dump(sol,open(sys.argv[2],'w'))
