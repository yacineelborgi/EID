"""Rigorous branching on the number of sources per type: for each leaf (K_t ranges), solve the LP
relaxation and evaluate the exact safe bound. UB = max over leaves (leaves cover all integers).

usage: python bb.py 'n:<=24' 'c:>=55' ...   (prints the safe bound of that leaf)
"""
import json
import sys
import time

import numpy as np

from relax import INF, TYPES, build, safe_bound, to_highs

inst = json.load(open("/home/user/EID/datasets/5_energy.json"))
M = inst["models"]
TAUS = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)


def leaf(conds, mdl=None):
    t0 = time.time()
    mdl = build(inst, TAUS)
    for c in conds:
        t, op = c.split(":")
        cs = [(j, 1) for i, j in mdl.sv.items() if t in M[i]["dataRequirements"]]
        if op.startswith("<="):
            mdl.row(cs, -INF, int(op[2:]), f"br{t}")
        else:
            mdl.row(cs, int(op[2:]), INF, f"br{t}")
    h, _ = to_highs(mdl, integer=False)
    h.run()
    st = h.getModelStatus()
    if str(st).endswith("kInfeasible"):
        return None, "infeasible"
    lp = h.getInfo().objective_function_value
    y = np.array(h.getSolution().row_dual)
    B, _ = safe_bound(mdl, y)
    if B is None or float(B) > 1e7:
        B, _ = safe_bound(mdl, -y)
    x = np.array(h.getSolution().col_value)
    K = {t: sum(x[j] for i, j in mdl.sv.items() if t in M[i]["dataRequirements"]) for t in TYPES}
    return B, f"LP {lp:.4f} safe {float(B):.4f} K {({t: round(v, 3) for t, v in K.items()})} ({time.time() - t0:.0f}s)"


if __name__ == "__main__":
    B, msg = leaf(sys.argv[1:])
    print(sys.argv[1:], msg, flush=True)
