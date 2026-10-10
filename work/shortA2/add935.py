import json, sys, itertools
sys.path.insert(0,'.'); sys.path.insert(0,'/home/user/EID')
import fix
I=json.load(open('/home/user/EID/datasets/6_shortage.json'));M=I['models'];D=I['datasets']
s=json.load(open(sys.argv[1])); X=int(sys.argv[2])
dm={d:m for d,m in s['dataMappings']}; mm=s['modelMappings']
src={a for a,b in mm}; tg={b for a,b in mm}
cop={m for d,m in dm.items() if D[d]['isCopyrighted']}
trained=(set(dm.values())-src)|tg
halfish=tg|cop|src
assert X not in trained
new=[]
for t,r in M[X]['dataRequirements'].items():
    L,U=r['lowerBound'],r['upperBound']
    pool=[d['id'] for d in D if d['dataType']==t and d['id'] not in dm]
    pf=sorted([d for d in pool if not D[d]['isCopyrighted']],key=lambda d:D[d]['size'])
    done=None
    for k in (1,2,3):
        for c in itertools.combinations(pf[:40],k):
            sm=sum(D[d]['size'] for d in c)
            if L<=sm<=U: done=list(c); break
        if done: break
    if not done:
        # swap: free item f in halfish bin -> replaced by pool items
        bins={}
        for d,m in dm.items():
            if D[d]['dataType']==t: bins.setdefault(m,[]).append(d)
        for m,ds in bins.items():
            if m not in halfish or done: continue
            rq=M[m]['dataRequirements'][t]; S=sum(D[d]['size'] for d in ds)
            for f in ds:
                if D[f]['isCopyrighted'] or not (L<=D[f]['size']<=U): continue
                rest=S-D[f]['size']
                for k in (0,1,2):
                    for c in itertools.combinations(pool[:60] if k==2 else pool,k):
                        sm=rest+sum(D[d]['size'] for d in c)
                        if rq['lowerBound']<=sm<=rq['upperBound']:
                            for d in c: dm[d]=m
                            del dm[f]; done=[f]; break
                    if done: break
                if done: break
    print(t,L,U,'->',done,[D[d]['size'] for d in (done or [])])
    if not done: sys.exit(0)
    for d in done: dm[d]=X
sol={'dataMappings':[[d,m] for d,m in dm.items()],'modelMappings':mm}
print('score',fix.check(sol)); print('saved',fix.save_if_better(sol))
json.dump(sol,open(sys.argv[3],'w'))
