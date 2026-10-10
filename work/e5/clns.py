"""Compact LNS: neighbourhood = models whose requirements all have lb > TAUMAX (so the tiny-bin
costs stay constant); residual capacities computed exactly from the fixed part (exact P and G).
MIP with 12 rows (free_t, tot_t, src_t).

usage: python clns.py start_plan out_plan iters nb tl margin_free margin_tot_n,t,i,c seed lp_pickle
"""
import json
import pickle
import random
import sys
import time

import numpy as np
import highspy
from scipy.sparse import coo_matrix

from planchk import G

inst = json.load(open("/home/user/EID/datasets/5_energy.json"))
M, D = inst["models"], inst["datasets"]
TYPES = "ntic"
TAUS = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)
TAUMAX = 600
Fs = {t: sorted(d["size"] for d in D if d["dataType"] == t and not d["isCopyrighted"]) for t in TYPES}
Cs = {t: sorted(d["size"] for d in D if d["dataType"] == t and d["isCopyrighted"]) for t in TYPES}
INF = highspy.kHighsInf


def bad(i, t):
    r = M[i]["dataRequirements"][t]
    return r["upperBound"] < r["lowerBound"]


def plan_mode(plan):
    srq = {}
    for i, t in plan["srcreq"]:
        srq[i] = srq.get(i, "") + t
    md = {}
    for i in plan["full"]: md[i] = "full"
    for i in plan["source"]: md[i] = "src"
    for i in plan["half"]: md[i] = "half" + "".join(t for t in M[i]["dataRequirements"] if t in srq.get(i, ""))
    return md


def value(md):
    return sum(M[i]["value"] if m == "full" else 0 if m == "src" else M[i]["value"] // 2 for i, m in md.items())


def residuals(md, fixed_set, mf, mt, taus=TAUS):
    """capacities left for the neighbourhood, given modes of models in fixed_set."""
    res = {}
    for t in TYPES:
        lA = [M[i]["dataRequirements"][t]["lowerBound"] for i in fixed_set if md.get(i) == "full" and t in M[i]["dataRequirements"]]
        lB = [M[i]["dataRequirements"][t]["lowerBound"] for i in fixed_set if md.get(i, "off") not in ("off", "full")
              and t in M[i]["dataRequirements"] and not bad(i, t) and t not in (md[i][4:] if md[i].startswith("half") else "")]
        fr, tr = [], []
        sF, sT = sum(Fs[t]), sum(Fs[t]) + sum(Cs[t])
        for tau in taus:
            kf = sum(1 for l in lA if l <= tau); kh = sum(1 for l in lB if l <= tau)
            fr.append(sF - mf - sum(l for l in lA if l > tau) - sum(Fs[t][:kf]))
            tr.append(sT - mt[t] - sum(l for l in lA + lB if l > tau) - G(Fs[t], Cs[t], kf, kh))
        src = sum(1 for i in fixed_set if md.get(i) == "src" and t in M[i]["dataRequirements"]) - \
            sum(1 for i in fixed_set if md.get(i, "off").startswith("half") and t in md[i][4:])
        res[t] = (min(fr), min(tr), src)
    return res


def solve_nb(N, md, res, tl, start=True):
    N = sorted(N)
    c, lo, up, cols = [], [], [], []
    rows, cs, vals = [], [], []
    var = {}
    def nv(key, cost):
        var[key] = len(c); c.append(cost); return var[key]
    r = 12  # 0-3 free, 4-7 tot, 8-11 src
    rlo, rhi = [-INF] * 12, [res[t][0] for t in TYPES] + [res[t][1] for t in TYPES] + [res[t][2] for t in TYPES]
    for i in N:
        m = M[i]; reqs = m["dataRequirements"]
        isbad = any(bad(i, t) for t in reqs)
        mono = len(reqs) == 1
        one = []
        if not isbad and m["value"] > 0:
            j = nv((i, "f"), m["value"]); one.append(j)
            for k, t in enumerate(TYPES):
                if t in reqs:
                    rows += [k, 4 + k]; cs += [j, j]; vals += [reqs[t]["lowerBound"]] * 2
        if m["value"] // 2 > 0:
            j = nv((i, "h"), m["value"] // 2); one.append(j)
            for k, t in enumerate(TYPES):
                if t in reqs:
                    zj = nv((i, "z", t), 0)
                    rows += [r, r]; cs += [zj, j]; vals += [1, -1]
                    rlo.append(0 if bad(i, t) else -INF); rhi.append(0); r += 1
                    rows.append(8 + k); cs.append(zj); vals.append(1)
                    if not bad(i, t):
                        rows += [4 + k, 4 + k]; cs += [j, zj]; vals += [reqs[t]["lowerBound"], -reqs[t]["lowerBound"]]
        if mono and not isbad:
            t = next(iter(reqs)); k = TYPES.index(t)
            j = nv((i, "s"), 0); one.append(j)
            rows += [4 + k, 8 + k]; cs += [j, j]; vals += [reqs[t]["lowerBound"], -1]
        if len(one) > 1:
            for j in one:
                rows.append(r); cs.append(j); vals.append(1)
            rlo.append(-INF); rhi.append(1); r += 1
    nV = len(c)
    A = coo_matrix((np.array(vals, float), (rows, cs)), shape=(r, nV)).tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(tl))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("mip_abs_gap", 0.99)
    lp = highspy.HighsLp()
    lp.num_col_ = nV; lp.num_row_ = r
    lp.col_cost_ = np.array(c, float); lp.col_lower_ = np.zeros(nV); lp.col_upper_ = np.ones(nV)
    lp.row_lower_ = np.array(rlo, float); lp.row_upper_ = np.array(rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.integrality_ = [highspy.HighsVarType.kInteger] * nV
    h.passModel(lp)
    if start:
        x = np.zeros(nV)
        for i in N:
            m = md.get(i, "off")
            if m == "full": x[var[(i, "f")]] = 1
            elif m == "src": x[var[(i, "s")]] = 1
            elif m.startswith("half"):
                x[var[(i, "h")]] = 1
                for t in m[4:]: x[var[(i, "z", t)]] = 1
        hs = highspy.HighsSolution(); hs.col_value = list(x); hs.value_valid = True
        h.setSolution(hs)
    h.run()
    info = h.getInfo()
    if info.primal_solution_status != 2:
        return None, info
    x = np.array(h.getSolution().col_value)
    new = {}
    for i in N:
        if (i, "f") in var and x[var[(i, "f")]] > 0.5: new[i] = "full"
        elif (i, "s") in var and x[var[(i, "s")]] > 0.5: new[i] = "src"
        elif (i, "h") in var and x[var[(i, "h")]] > 0.5:
            new[i] = "half" + "".join(t for t in M[i]["dataRequirements"] if x[var[(i, "z", t)]] > 0.5)
        else: new[i] = "off"
    return new, info


def to_plan(md):
    return {"full": [i for i, m in md.items() if m == "full"],
            "half": [i for i, m in md.items() if m.startswith("half")],
            "source": [i for i, m in md.items() if m == "src"],
            "srcreq": [[i, t] for i, m in md.items() if m.startswith("half") for t in m[4:]],
            "obj": value(md)}


def main():
    a = sys.argv
    md = plan_mode(json.load(open(a[1]))); out = a[2]
    iters, nb, tl = int(a[3]), int(a[4]), float(a[5])
    mf = int(a[6]); mt = dict(zip(TYPES, [int(x) for x in a[7].split(",")]))
    rng = random.Random(int(a[8]))
    pk = pickle.load(open(a[9], "rb"))
    modes = pk["modes"]
    sec = {i: (sorted(float(v) for v in d.values())[1] if len(d) > 1 else 1e9) for i, d in modes.items()}
    elig = [i for i in range(len(M)) if all(r["lowerBound"] > TAUMAX for r in M[i]["dataRequirements"].values())]
    cand = [i for i in elig if sec[i] <= 25]
    print("eligible", len(elig), "cand", len(cand), flush=True)
    best = value(md)
    allm = set(range(len(M)))
    # check start feasibility
    res0 = residuals(md, allm, mf, mt)
    print("start", best, "residuals", res0, flush=True)
    t0 = time.time()
    for it in range(iters):
        N = set(rng.sample(cand, min(nb, len(cand))))
        res = residuals(md, allm - N, mf, mt, taus=tuple(x for x in TAUS if x <= TAUMAX))
        new, info = solve_nb(N, md, res, tl)
        if new is None:
            print(" it", it, "no sol", flush=True); continue
        md2 = dict(md); md2.update(new)
        md2 = {i: m for i, m in md2.items() if m != "off"}
        v = value(md2)
        print(f" it {it}: obj {info.objective_function_value:.0f} bound {info.mip_dual_bound:.1f} val {v} best {best} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if v > best:
            chk = residuals(md2, allm, mf, mt)
            if all(chk[t][0] >= 0 and chk[t][1] >= 0 and chk[t][2] >= 0 for t in TYPES):
                best = v; md = md2
                json.dump(to_plan(md), open(out, "w"))
            else:
                print("  infeasible?", chk, flush=True)


if __name__ == "__main__":
    main()
