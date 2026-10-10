import json
import pickle
import sys
import time

import numpy as np

from relax import INF, build, safe_bound, to_highs

inst = json.load(open("/home/user/EID/datasets/5_energy.json"))
taus = tuple(int(x) for x in sys.argv[1].split(","))
t0 = time.time()
mdl = build(inst, taus)
h, A = to_highs(mdl, integer=False)
h.run()
sol = h.getSolution()
y = np.array(sol.row_dual)
B, red = safe_bound(mdl, y)
if B is None:
    y = -y
    B, red = safe_bound(mdl, y)
x = np.array(sol.col_value)
print("U", float(B), f"{time.time() - t0:.0f}s")
r = np.array([float(v) for v in red])
integ = np.array(mdl.integ, bool)
ar = np.abs(r[integ])
for d in (5, 10, 20, 50, 100, 150, 200, 300):
    print(d, int((ar <= d).sum()))
frac = [(mdl.names[j], x[j]) for j in range(len(x)) if integ[j] and 1e-6 < x[j] < 1 - 1e-6]
print("fractional", len(frac), frac[:40])
pickle.dump({"U": B, "red": red, "x": x, "y": y, "taus": taus}, open(f"lp_{sys.argv[1].replace(',', '_')}.pkl", "wb"))
