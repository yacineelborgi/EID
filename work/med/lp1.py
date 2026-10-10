import sys, json, pickle
sys.path.insert(0,'.')
from rel import *
inst = load('/home/user/EID/datasets/2_medium.json')
R = Rel(inst, tau={t:1 for t in TYPES}); R.build()
out = R.solve()
u = R.multipliers_from_duals(out['dual'])
for n,v in u.items():
    if v: print(n, float(v))
L, red = R.lagrangian(u)
print('L', float(L))
x = out['x']; nM=R.nM
frac = [(k, x[k]) for k in range(R.nV) if 1e-6 < x[k] < 1-1e-6]
print('fractional', len(frac), frac[:20])
# deviation distribution
devs=[]
for i,ops in enumerate(red):
    best=max(ops)[0]
    for p,name in ops:
        if p<best: devs.append((float(best-p), i, name))
devs.sort()
import numpy as np
dc=np.array([d[0] for d in devs])
for th in [1,2,5,10,20,31]:
    print(th, (dc<=th).sum(), len({d[1] for d in devs if d[0]<=th}))
pickle.dump(dict(u=u,L=L,red=red,x=x), open('lp1.pkl','wb'))
