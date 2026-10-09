"""Phase 1: choose which models to train and in which mode, with a MIP.

Modes per model:
  full   -> every requirement filled with free datasets       (value v)
  half   -> requirements filled with any data and/or sources   (value v//2)
  source -> mono-requirement model used as a data generator    (value 0)

Data is aggregated per type (the actual dataset assignment is done in assign.py),
so the MIP only sees "amount of data consumed" = lowerBound (+ a small margin).
"""
import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

TYPES = "ntic"


def load(path):
    with open(path) as f:
        return json.load(f)


def build_and_solve(inst, alpha_free=1.0, alpha_tot=1.0, extra=0.0, time_limit=120,
                    forced_off=(), gap=1e-5, verbose=False):
    M = inst["models"]
    D = inst["datasets"]
    cap = inst["energyCap"]
    nM = len(M)
    F = {t: 0 for t in TYPES}
    T = {t: 0 for t in TYPES}
    for d in D:
        T[d["dataType"]] += d["size"]
        if not d["isCopyrighted"]:
            F[d["dataType"]] += d["size"]

    bad = np.zeros(nM, bool)
    mono = np.zeros(nM, bool)
    for i, m in enumerate(M):
        reqs = m["dataRequirements"]
        bad[i] = any(r["upperBound"] < r["lowerBound"] for r in reqs.values())
        mono[i] = len(reqs) == 1
    off = np.zeros(nM, bool)
    off[list(forced_off)] = True

    zlist = []  # (model, type, lb, isbad)
    for i, m in enumerate(M):
        for t, r in m["dataRequirements"].items():
            zlist.append((i, t, r["lowerBound"], r["upperBound"] < r["lowerBound"]))
    nZ = len(zlist)
    nV = 3 * nM + nZ
    fo, ho, so, zo = 0, nM, 2 * nM, 3 * nM

    c = np.zeros(nV)
    for i, m in enumerate(M):
        c[fo + i] = -m["value"]
        c[ho + i] = -(m["value"] // 2)
    # tiny penalty on sources to avoid useless ones
    c[so:so + nM] = 1e-6

    lbv = np.zeros(nV)
    ubv = np.ones(nV)
    ubv[fo:fo + nM][bad | off] = 0
    ubv[ho:ho + nM][off] = 0
    ubv[so:so + nM][(~mono) | bad | off] = 0

    rows, cols, vals = [], [], []
    rlo, rhi = [], []
    r = 0
    # 1. f+h+s <= 1
    for i in range(nM):
        rows += [r, r, r]; cols += [fo + i, ho + i, so + i]; vals += [1, 1, 1]
        rlo.append(-np.inf); rhi.append(1); r += 1
    # 2. z <= h (== h for bad requirement)
    for k, (i, t, lb, isb) in enumerate(zlist):
        rows += [r, r]; cols += [zo + k, ho + i]; vals += [1, -1]
        rlo.append(0 if isb else -np.inf); rhi.append(0); r += 1
    # 3. free data per type, 4. total data per type, 5. sources per type
    rfree = {t: r + j for j, t in enumerate(TYPES)}; r += 4
    rtot = {t: r + j for j, t in enumerate(TYPES)}; r += 4
    rsrc = {t: r + j for j, t in enumerate(TYPES)}; r += 4
    for t in TYPES:
        rlo += [-np.inf]; rhi += [alpha_free]
    for t in TYPES:
        rlo += [-np.inf]; rhi += [alpha_tot]
    for t in TYPES:
        rlo += [-np.inf]; rhi += [0]
    for k, (i, t, lb, isb) in enumerate(zlist):
        need = (lb + extra) / max(F[t], 1)
        if not isb:
            rows.append(rfree[t]); cols.append(fo + i); vals.append(need)
            rows.append(rtot[t]); cols.append(fo + i); vals.append((lb + extra) / T[t])
            rows.append(rtot[t]); cols.append(ho + i); vals.append((lb + extra) / T[t])
            rows.append(rtot[t]); cols.append(zo + k); vals.append(-(lb + extra) / T[t])
        rows.append(rsrc[t]); cols.append(zo + k); vals.append(1)
    for i, m in enumerate(M):
        if mono[i] and not bad[i]:
            t, rr = next(iter(m["dataRequirements"].items()))
            rows.append(rtot[t]); cols.append(so + i); vals.append((rr["lowerBound"] + extra) / T[t])
            rows.append(rsrc[t]); cols.append(so + i); vals.append(-1)
    # 6. energy
    for i, m in enumerate(M):
        e = m["energyCost"] / cap
        for o in (fo, ho, so):
            rows.append(r); cols.append(o + i); vals.append(e)
    rlo.append(-np.inf); rhi.append(1.0); r += 1

    A = coo_matrix((vals, (rows, cols)), shape=(r, nV)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, rlo, rhi), integrality=np.ones(nV),
               bounds=Bounds(lbv, ubv),
               options={"time_limit": time_limit, "disp": verbose, "mip_rel_gap": gap})
    x = np.round(res.x).astype(int)
    plan = {"full": [], "half": [], "source": [], "srcreq": []}
    for i in range(nM):
        if x[fo + i]: plan["full"].append(i)
        if x[ho + i]: plan["half"].append(i)
        if x[so + i]: plan["source"].append(i)
    for k, (i, t, lb, isb) in enumerate(zlist):
        if x[zo + k]:
            plan["srcreq"].append([i, t])
    plan["obj"] = -float(res.fun)
    plan["bound"] = -float(getattr(res, "mip_dual_bound", res.fun) or res.fun)
    return plan


if __name__ == "__main__":
    inst = load(sys.argv[1])
    p = build_and_solve(inst, time_limit=float(sys.argv[2]) if len(sys.argv) > 2 else 60,
                        verbose=False)
    print("obj", p["obj"], "bound", p["bound"], "full", len(p["full"]), "half", len(p["half"]),
          "src", len(p["source"]), "srcreq", len(p["srcreq"]))
