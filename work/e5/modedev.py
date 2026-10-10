"""Per-model Lagrangian mode deviations (exact Fractions) from saved LP duals.
Coupling rows dualized: everything except one_i / zh_it rows (kept per model)."""
import json
import pickle
import sys
from fractions import Fraction

import numpy as np

from relax import INF, TYPES, build

inst = json.load(open("/home/user/EID/datasets/5_energy.json"))
M = inst["models"]


def mode_devs(mdl, y, lo=None, up=None):
    lo = mdl.lo if lo is None else lo
    up = mdl.up if up is None else up
    nR = len(mdl.rlo)
    Y = []
    for r in range(nR):
        nm = mdl.rnames[r]
        if nm.startswith("one") or nm.startswith("zh"):
            Y.append(Fraction(0)); continue
        yr = Fraction(float(y[r]))
        if mdl.rhi[r] == INF and yr > 0: yr = Fraction(0)
        if mdl.rlo[r] == -INF and yr < 0: yr = Fraction(0)
        Y.append(yr)
    red = [Fraction(int(c)) for c in mdl.c]
    for r, j, v in zip(mdl.rows, mdl.cols, mdl.vals):
        if Y[r]:
            red[j] -= Y[r] * int(v)
    U = Fraction(0)
    for r in range(nR):
        if Y[r] > 0: U += Y[r] * int(mdl.rhi[r])
        elif Y[r] < 0: U += Y[r] * int(mdl.rlo[r])
    # continuous vars
    modelvars = set(mdl.fv.values()) | set(mdl.hv.values()) | set(mdl.sv.values()) | set(mdl.zv.values())
    for j in range(len(red)):
        if j in modelvars: continue
        if red[j] > 0:
            assert up[j] != INF, mdl.names[j]
            U += red[j] * int(up[j])
        elif red[j] < 0:
            U += red[j] * int(lo[j])
    modes = {}
    for i, m in enumerate(M):
        opts = {"off": Fraction(0)}
        if i in mdl.fv and up[mdl.fv[i]] > 0:
            opts["full"] = red[mdl.fv[i]]
        if i in mdl.sv and up[mdl.sv[i]] > 0:
            opts["src"] = red[mdl.sv[i]]
        if i in mdl.hv and up[mdl.hv[i]] > 0:
            # enumerate sourced subsets
            ts = list(m["dataRequirements"])
            bad = [t for t in ts if m["dataRequirements"][t]["upperBound"] < m["dataRequirements"][t]["lowerBound"]]
            for mask in range(1 << len(ts)):
                Z = [ts[k] for k in range(len(ts)) if mask >> k & 1]
                if any(t not in Z for t in bad): continue
                if any(up[mdl.zv[(i, t)]] == 0 for t in Z): continue
                opts["half" + "".join(Z)] = red[mdl.hv[i]] + sum(red[mdl.zv[(i, t)]] for t in Z)
        # forced lower bounds (force_full etc.)
        if i in mdl.fv and lo[mdl.fv[i]] > 0:
            opts = {"full": opts["full"]}
        best = max(opts.values())
        U += best
        modes[i] = {k: best - v for k, v in opts.items()}
    return U, modes, red


if __name__ == "__main__":
    pk = pickle.load(open(sys.argv[1], "rb"))
    mdl = build(inst, pk["taus"])
    U, modes, red = mode_devs(mdl, pk["y"])
    print("U", float(U))
    sec = []
    for i, d in modes.items():
        vals = sorted(d.values())
        sec.append(vals[1] if len(vals) > 1 else Fraction(10**9))
    sec = np.array([float(v) for v in sec])
    for D in (1, 5, 10, 20, 50, 100, 150, 200):
        print(D, int((sec <= D).sum()))
    pickle.dump({"U": U, "modes": modes}, open(sys.argv[1].replace(".pkl", "_modes.pkl"), "wb"))
