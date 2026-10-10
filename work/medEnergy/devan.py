import sys, json, pickle
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/e5")
from relax import build, to_highs, INF
from modedev import mode_devs
from rmip5 import plan_mode
taus = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)
inst = json.load(open("/home/user/EID/datasets/5_energy.json")); M = inst["models"]
mf, mt = int(sys.argv[2]), int(sys.argv[3])
mdl = build(inst, taus)
for r, nm in enumerate(mdl.rnames):
    if nm.startswith("free"): mdl.rhi[r] -= mf
    if nm.startswith("tot"): mdl.rhi[r] -= mt
h, _ = to_highs(mdl, integer=False); h.run()
y = np.array(h.getSolution().row_dual); x = np.array(h.getSolution().col_value)
U, modes, red = mode_devs(mdl, y)
if U > 1e9: U, modes, red = mode_devs(mdl, -y); y = -y
pickle.dump(dict(U=U, modes=modes, y=y, x=x), open('lp5_%d_%d.pkl' % (mf, mt), 'wb'))
plan = json.load(open(sys.argv[1])); pm = plan_mode(plan)
devs = []
for i, d in modes.items():
    md = pm.get(i, 'off')
    if md not in d: print('mode missing', i, md, d.keys()); continue
    devs.append((float(d[md]), i, md, min(d, key=d.get)))
devs.sort(reverse=True)
print('U', float(U), 'sum dev', sum(v for v, *_ in devs))
print('top devs', [(round(v, 2), i, md, bm, M[i]['value']) for v, i, md, bm in devs[:25]])
for r, nm in enumerate(mdl.rnames):
    if abs(y[r]) > 1e-9 and not nm.startswith(('one', 'zh')): print(nm, 'dual', y[r])
