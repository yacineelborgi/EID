import sys, time, json, pickle
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
import numpy as np
from lp7 import Inst
from mip7 import Model, cut_set, profits, plan_from_x, plan_value, plan_energy, matching_lb
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--thr", type=float, default=3.0)
ap.add_argument("--tl", type=float, default=300)
ap.add_argument("--margin", type=str, default="150,150,150,150")
ap.add_argument("--emargin", type=float, default=2e-7)
ap.add_argument("--hw", type=int, default=30)
ap.add_argument("--out", type=str, default="plan_a.json")
ap.add_argument("--start", type=str, default="")
a = ap.parse_args()
margin = [int(x) for x in a.margin.split(",")]
I = Inst()
centers = [382, 405, 401, 389]
cuts = cut_set(I, centers, a.hw)
print("cuts per type", [len(cuts[t]) for t in range(4)], flush=True)
t0 = time.time()
Mo = Model(I, cuts, margin, a.emargin)
r = Mo.solve()
print(f"LP {r['obj']:.4f} {r['status']} {time.time()-t0:.0f}s", flush=True)
y = Mo.multipliers(r["dual"])
pickle.dump((cuts, y, r["obj"]), open(a.out + ".lp.pkl", "wb"))
P = profits(I, y)
best = P.max(1); bo = P.argmax(1)
dev = best[:, None] - P
nM = I.nM
lo = Mo.lo.copy(); up = Mo.up.copy()
free_models = np.nonzero(((dev > 1e-9) & (dev <= a.thr)).any(1) | ((dev <= 1e-9).sum(1) > 1))[0]
fixed = np.ones(nM, bool); fixed[free_models] = False
for o, off in ((1, 0), (2, nM), (3, 2 * nM)):
    col = np.arange(nM) + off
    # fixed models: option o at 1 if best, else 0
    m1 = fixed & (bo == o); m0 = fixed & (bo != o)
    lo[col[m1]] = 1; up[col[m0]] = 0
    # free models: forbid options with dev > thr
    mf = (~fixed) & (dev[:, o] > a.thr)
    up[col[mf]] = 0
print("free models", len(free_models), flush=True)
start = None
t0 = time.time()
rm = Mo.solve(integer=True, time_limit=a.tl, lo=lo, up=up)
print(f"MIP {rm['obj']:.2f} bound {rm['bound']:.2f} {rm['status']} {time.time()-t0:.0f}s", flush=True)
p = plan_from_x(I, rm["x"])
p["obj"] = rm["obj"]
E = plan_energy(I, p)
print("plan value", plan_value(I, p), "energy", E, "cap", I.cap, "over", E - I.cap, flush=True)
for t, (sl, ml) in enumerate(matching_lb(I, p)):
    print("  type", t, "sum lb", sl, "match LB", ml, "F", I.F[t], "slack", I.F[t] - ml, flush=True)
json.dump(p, open(a.out, "w"))
