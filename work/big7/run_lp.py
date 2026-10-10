import sys, time, pickle
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
import numpy as np
from lp7 import Inst, LP, separate, exact_bound
I = Inst()
print("F", I.F, "T", I.T, flush=True)
cuts = {t: [] for t in range(4)}
for it in range(int(sys.argv[1]) if len(sys.argv) > 1 else 8):
    t0 = time.time()
    L = LP(I, cuts)
    r = L.solve()
    f = r["x"][:I.nM]
    print(f"it {it} LP {r['obj']:.4f} {r['status']} {time.time()-t0:.0f}s ncuts {[len(cuts[t]) for t in range(4)]}", flush=True)
    new = 0
    for t in range(4):
        sep = separate(I, f, t, top=3)
        print("  t", t, sep[:3], flush=True)
        for j, vv in sep:
            if j not in cuts[t] and j > 0:
                cuts[t].append(j); new += 1
        cuts[t].sort()
    y = L.multipliers(r["dual"])
    pickle.dump((cuts, y, r["obj"]), open("lp_state.pkl", "wb"))
    if new == 0:
        break
t0 = time.time()
B, bo = exact_bound(I, y)
print("exact bound", float(B), "floor", B.numerator // B.denominator, f"{time.time()-t0:.0f}s", flush=True)
