import sys, pickle
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
import numpy as np
from lp7 import Inst
I = Inst()
cuts, y, obj = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "lp_state.pkl", "rb"))
def profits(I, y):
    nM = I.nM
    g = np.zeros((nM, 4))
    for t in range(4):
        lbt = I.lb[:, t].astype(float)
        u = (y[("F", t)] + y[("T", t)]) * lbt
        for j, c in y[("C", t)]:
            if c > 0: u += c * np.maximum(lbt, float(I.fs[t][j]))
        g[:, t] = np.where(I.has[:, t], u, 0)
    base = y["E"] * I.e.astype(float)
    pf = I.v - base - g.sum(1); pf[I.bad | (I.v <= 0)] = -np.inf
    hu = np.zeros(nM)
    for t in range(4):
        lbt = I.lb[:, t].astype(float)
        c = np.where(I.badt[:, t], y[("S", t)], np.minimum(y[("T", t)] * lbt, y[("S", t)]))
        hu += np.where(I.has[:, t], c, 0)
    ph = (I.v // 2) - base - hu; ph[(I.v // 2) <= 0] = -np.inf
    ps = np.full(nM, -np.inf)
    m = I.mono & ~I.bad
    t = I.mtype
    ps[m] = (np.array([y[("S", tt)] for tt in t]) - base - np.array([y[("T", tt)] for tt in t]) * I.lb[np.arange(nM), t])[m]
    return np.stack([np.zeros(nM), pf, ph, ps], 1)
P = profits(I, y)
best = P.max(1)
print("sum best", best.sum())
dev = best[:, None] - P
d = np.sort(dev[np.isfinite(dev) & (dev > 1e-12)])
for th in [0.1, 0.5, 1, 2, 3, 5, 10, 20]:
    print(th, int((d <= th).sum()))
bo = P.argmax(1)
print("best options counts", np.bincount(bo, minlength=4))
print("near-zero (<1e-6) deviations", int((dev[np.isfinite(dev)] <= 1e-6).sum()) - I.nM)
