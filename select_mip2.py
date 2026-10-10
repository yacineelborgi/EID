"""MIP selection aware of the discreteness of small datasets.

Same modes as select_mip.py (full / half / source), but requirements with a small
lower bound (< SMALL_L) are not counted at `lowerBound`: every bin needs at least
one dataset, so k small bins of a type consume at least the k smallest datasets.
That convex piecewise-linear lower bound S(k) is added with tangent cuts, once for
the free pool (full models) and once for all datasets of the type.
"""
import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

TYPES = "ntic"
SC = 1e-4  # data scaling


def build_and_solve(inst, alpha_free=1.0, alpha_tot=1.0, small_L=2500, time_limit=120,
                    forced_off=(), gap=1e-7, verbose=False, resid_items=None, cap=None,
                    max_cuts=600, energy_alpha=1.0, forced_on=None):
    """resid_items: optional {t: (free_sizes, copy_sizes)} instead of the instance datasets.
    forced_on: optional dict {model: mode} fixing modes (mode in 'full','half','source')."""
    M = inst["models"]
    D = inst["datasets"]
    cap = inst["energyCap"] if cap is None else cap
    nM = len(M)
    if resid_items is None:
        fs = {t: sorted(d["size"] for d in D if d["dataType"] == t and not d["isCopyrighted"]) for t in TYPES}
        cs = {t: sorted(d["size"] for d in D if d["dataType"] == t and d["isCopyrighted"]) for t in TYPES}
    else:
        fs = {t: sorted(resid_items[t][0]) for t in TYPES}
        cs = {t: sorted(resid_items[t][1]) for t in TYPES}
    alls = {t: sorted(fs[t] + cs[t]) for t in TYPES}
    F = {t: sum(fs[t]) for t in TYPES}
    T = {t: sum(alls[t]) for t in TYPES}

    bad = np.zeros(nM, bool)
    mono = np.zeros(nM, bool)
    for i, m in enumerate(M):
        reqs = m["dataRequirements"]
        bad[i] = any(r["upperBound"] < r["lowerBound"] for r in reqs.values())
        mono[i] = len(reqs) == 1
    off = np.zeros(nM, bool)
    off[list(forced_off)] = True

    zlist = []
    for i, m in enumerate(M):
        for t, r in m["dataRequirements"].items():
            zlist.append((i, t, r["lowerBound"], r["upperBound"] < r["lowerBound"]))
    nZ = len(zlist)
    fo, ho, so, zo = 0, nM, 2 * nM, 3 * nM
    co = 3 * nM + nZ  # CF_t (4), CT_t (4), KF_t (4), KA_t (4)
    nV = co + 16

    c = np.zeros(nV)
    for i, m in enumerate(M):
        c[fo + i] = -m["value"]
        c[ho + i] = -(m["value"] // 2)
    c[so:so + nM] = 1e-6
    lbv = np.zeros(nV)
    ubv = np.ones(nV)
    ubv[co:] = np.inf
    ubv[fo:fo + nM][bad | off] = 0
    ubv[ho:ho + nM][off] = 0
    ubv[so:so + nM][(~mono) | bad | off] = 0
    if forced_on:
        for i, mode in forced_on.items():
            o = {"full": fo, "half": ho, "source": so}[mode]
            lbv[o + i] = 1

    rows, cols, vals, rlo, rhi = [], [], [], [], []
    r = 0

    def add_row(entries, lo, hi):
        nonlocal r
        for j, v in entries:
            rows.append(r); cols.append(j); vals.append(v)
        rlo.append(lo); rhi.append(hi); r += 1

    for i in range(nM):
        add_row([(fo + i, 1), (ho + i, 1), (so + i, 1)], -np.inf, 1)
    for k, (i, t, lb, isb) in enumerate(zlist):
        add_row([(zo + k, 1), (ho + i, -1)], 0 if isb else -np.inf, 0)

    # per-type rows
    free_row = {t: [] for t in TYPES}   # big full bins
    tot_row = {t: [] for t in TYPES}    # big bins, any mode
    cntF = {t: [] for t in TYPES}       # count of small full bins
    cntA = {t: [] for t in TYPES}       # count of small bins (all modes)
    lbF = {t: [] for t in TYPES}        # lb of small full bins
    lbA = {t: [] for t in TYPES}
    src_row = {t: [] for t in TYPES}
    for k, (i, t, lb, isb) in enumerate(zlist):
        src_row[t].append((zo + k, 1))
        if isb:
            continue
        small = lb < small_L
        if small:
            cntF[t].append((fo + i, 1)); cntA[t] += [(fo + i, 1), (ho + i, 1), (zo + k, -1)]
            lbF[t].append((fo + i, lb * SC)); lbA[t] += [(fo + i, lb * SC), (ho + i, lb * SC), (zo + k, -lb * SC)]
        else:
            free_row[t].append((fo + i, lb * SC))
            tot_row[t] += [(fo + i, lb * SC), (ho + i, lb * SC), (zo + k, -lb * SC)]
    for i, m in enumerate(M):
        if mono[i] and not bad[i]:
            t, rr = next(iter(m["dataRequirements"].items()))
            lb = rr["lowerBound"]
            src_row[t].append((so + i, -1))
            if lb < small_L:
                cntA[t].append((so + i, 1)); lbA[t].append((so + i, lb * SC))
            else:
                tot_row[t].append((so + i, lb * SC))
    for j, t in enumerate(TYPES):
        CF, CT, KF, KA = co + j, co + 4 + j, co + 8 + j, co + 12 + j
        add_row([(KF, 1)] + [(v, -x) for v, x in cntF[t]], 0, 0)
        add_row([(KA, 1)] + [(v, -x) for v, x in cntA[t]], 0, 0)
        add_row(free_row[t] + [(CF, 1)], -np.inf, alpha_free * F[t] * SC)
        add_row(tot_row[t] + [(CT, 1)], -np.inf, alpha_tot * T[t] * SC)
        add_row(src_row[t], -np.inf, 0)
        # CF >= sum lb small full ; CT >= sum lb small all
        add_row([(CF, 1)] + [(v, -x) for v, x in lbF[t]], 0, np.inf)
        add_row([(CT, 1)] + [(v, -x) for v, x in lbA[t]], 0, np.inf)
        # tangent cuts: C >= S(j) + s_{j+1} (k - j)
        for sizes, K, C in ((fs[t], KF, CF), (alls[t], KA, CT)):
            S = 0
            for jj in range(min(len(sizes), max_cuts)):
                sj1 = sizes[jj]
                # C - sj1*k >= S - jj*sj1
                add_row([(C, 1), (K, -sj1 * SC)], (S - jj * sj1) * SC, np.inf)
                S += sj1
            # cannot have more small bins than items
            add_row([(K, 1)], -np.inf, len(sizes))
    # energy
    ent = []
    for i, m in enumerate(M):
        e = m["energyCost"] / cap
        ent += [(fo + i, e), (ho + i, e), (so + i, e)]
    add_row(ent, -np.inf, energy_alpha)

    A = coo_matrix((vals, (rows, cols)), shape=(r, nV)).tocsr()
    integ = np.ones(nV)
    integ[co:] = 0
    res = milp(c, constraints=LinearConstraint(A, rlo, rhi), integrality=integ,
               bounds=Bounds(lbv, ubv),
               options={"time_limit": time_limit, "disp": verbose, "mip_rel_gap": gap})
    if res.x is None:
        return None
    x = np.round(res.x[:co]).astype(int)
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
    plan["CF"] = [float(v) / SC for v in res.x[co:co + 4]]
    plan["CT"] = [float(v) / SC for v in res.x[co + 4:co + 8]]
    return plan


if __name__ == "__main__":
    import os, time
    path = sys.argv[1]
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    a = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    L = int(sys.argv[4]) if len(sys.argv) > 4 else 2500
    inst = json.load(open(path))
    t0 = time.time()
    p = build_and_solve(inst, alpha_free=a, alpha_tot=a, small_L=L, time_limit=tl, verbose=True)
    print("time", time.time() - t0, "obj", p["obj"], "bound", p["bound"], "full", len(p["full"]), "half", len(p["half"]),
          "src", len(p["source"]), "CF", p["CF"], "CT", p["CT"], flush=True)
    os.makedirs("plans", exist_ok=True)
    out = f"plans/{os.path.basename(path)[:-5]}_s2_{a}_{L}.json"
    with open(out, "w") as f:
        json.dump(p, f)
    print("wrote", out)
