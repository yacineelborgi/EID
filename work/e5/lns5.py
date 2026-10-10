"""LNS on the rigorous relaxation (with margins): fix all models to the incumbent except a
neighbourhood (random subset of near-zero reduced-cost models + models off their LP mode), MIP.

usage: python lns5.py start_plan out_plan iters nb tl margin_free margin_tot_n,t,i,c [seed]
"""
import json
import random
import sys
import time

import numpy as np
import highspy

from relax import INF, TYPES, build, to_highs
from modedev import mode_devs
from rmip5 import plan_mode, set_mode

inst = json.load(open("/home/user/EID/datasets/5_energy.json"))
M = inst["models"]
TAUS = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)


def main():
    a = sys.argv
    start = json.load(open(a[1])); out = a[2]
    iters, nb, tl = int(a[3]), int(a[4]), float(a[5])
    mf = int(a[6]); mt = dict(zip(TYPES, [int(x) for x in a[7].split(",")]))
    seed = int(a[8]) if len(a) > 8 else 0
    rng = random.Random(seed)
    t0 = time.time()
    mdl = build(inst, TAUS)
    for r, nm in enumerate(mdl.rnames):
        if nm.startswith("free"): mdl.rhi[r] -= mf
        if nm.startswith("tot"): mdl.rhi[r] -= mt[nm[3]]
    h, _ = to_highs(mdl, integer=False)
    h.run()
    sol = h.getSolution()
    y = np.array(sol.row_dual)
    U, modes, red = mode_devs(mdl, y)
    if U > 1e9:
        U, modes, red = mode_devs(mdl, -y)
    lpobj = h.getInfo().objective_function_value
    print(f"LP {lpobj:.3f} U {float(U):.3f} ({time.time() - t0:.0f}s)", flush=True)
    sec = {}
    lpbest = {}
    for i, d in modes.items():
        vals = sorted(d.values())
        sec[i] = float(vals[1]) if len(vals) > 1 else 1e9
        lpbest[i] = min(d, key=d.get)
    cand = [i for i in modes if sec[i] <= 20]
    print("candidates", len(cand), flush=True)
    # prune tangent rows (per group) around LP active piece
    act = np.array(sol.row_value)
    keep = np.ones(len(mdl.rlo), bool)
    groups = {}
    for r, nm in enumerate(mdl.rnames):
        if nm[:2] in ("tF", "tA", "tB") and "_" in nm:
            g, j = nm.rsplit("_", 1)
            groups.setdefault(g, []).append((int(j), r))
    for g, lst in groups.items():
        jb = min((act[r] - mdl.rlo[r], j) for j, r in lst)[1]
        for j, r in lst:
            if abs(j - jb) > 100:
                keep[r] = False
    rm = -np.ones(len(mdl.rlo), int); rm[keep] = np.arange(keep.sum())
    m2 = type(mdl)()
    m2.c, m2.lo, m2.up, m2.integ, m2.names = mdl.c, list(mdl.lo), list(mdl.up), mdl.integ, mdl.names
    R = np.array(mdl.rows); sel = keep[R]
    m2.rows = list(rm[R[sel]]); m2.cols = list(np.array(mdl.cols)[sel]); m2.vals = list(np.array(mdl.vals)[sel])
    m2.rlo = [v for v, k in zip(mdl.rlo, keep) if k]; m2.rhi = [v for v, k in zip(mdl.rhi, keep) if k]
    m2.rnames = [v for v, k in zip(mdl.rnames, keep) if k]
    h2, _ = to_highs(m2, integer=True)
    h2.setOptionValue("mip_rel_gap", 0.0)
    h2.setOptionValue("mip_abs_gap", 0.99)
    h2.setOptionValue("time_limit", tl)
    nV = len(mdl.c)
    inc = plan_mode(start)
    D = inst["datasets"]
    Fs = {t: sorted(d["size"] for d in D if d["dataType"] == t and not d["isCopyrighted"]) for t in TYPES}
    Cs = {t: sorted(d["size"] for d in D if d["dataType"] == t and d["isCopyrighted"]) for t in TYPES}
    name2j = {nm: j for j, nm in enumerate(mdl.names)}
    def modes_to_x(md):
        x = np.zeros(nV)
        for i, m in md.items():
            if m == "full": x[mdl.fv[i]] = 1
            elif m == "src": x[mdl.sv[i]] = 1
            else:
                x[mdl.hv[i]] = 1
                for t in m[4:]: x[mdl.zv[(i, t)]] = 1
        for t in TYPES:
            lA = [M[i]["dataRequirements"][t]["lowerBound"] for i, m in md.items() if m == "full" and t in M[i]["dataRequirements"]]
            lB = [M[i]["dataRequirements"][t]["lowerBound"] for i, m in md.items() if m != "full" and t in M[i]["dataRequirements"]
                  and t not in (m[4:] if m.startswith("half") else "")
                  and M[i]["dataRequirements"][t]["upperBound"] >= M[i]["dataRequirements"][t]["lowerBound"]]
            for tau in TAUS:
                kf = sum(1 for l in lA if l <= tau); kh = sum(1 for l in lB if l <= tau)
                x[name2j[f"wF{t}{tau}"]] = sum(Fs[t][:kf])
                if f"a{t}{tau}" in name2j:
                    rest = sorted([(s, 0) for s in Fs[t][kf:kf + kh]] + [(s, 1) for s in Cs[t][:kh]])[:kh]
                    na = kf + sum(1 for s, c in rest if c == 0)
                    x[name2j[f"a{t}{tau}"]] = na; x[name2j[f"b{t}{tau}"]] = kf + kh - na
                    x[name2j[f"wA{t}{tau}"]] = sum(Fs[t][:na]); x[name2j[f"wB{t}{tau}"]] = sum(Cs[t][:kf + kh - na])
        return x
    def val(md):
        return sum(M[i]["value"] if m == "full" else 0 if m == "src" else M[i]["value"] // 2 for i, m in md.items())
    best = val(inc)
    print("start value", best, flush=True)
    for it in range(iters):
        diff = [i for i in cand if inc.get(i, "off") != lpbest[i]]
        N = set(rng.sample(diff, min(len(diff), nb // 3)))
        rest = [i for i in cand if i not in N]
        N |= set(rng.sample(rest, min(len(rest), nb - len(N))))
        lo = list(mdl.lo); up = list(mdl.up)
        for i in modes:
            if i in N: continue
            set_mode(mdl, lo, up, i, inc.get(i, "off"))
        h2.changeColsBounds(nV, np.arange(nV, dtype=np.int32), np.array(lo, float), np.array(up, float))
        hs = highspy.HighsSolution()
        # continuous vars: let HiGHS complete them -> provide only integers? give full LP-free vector
        xs = modes_to_x(inc)
        hs.col_value = list(xs)
        hs.value_valid = True
        r0 = h2.setSolution(hs)
        h2.run()
        info = h2.getInfo()
        st = h2.getModelStatus()
        if info.primal_solution_status != 2:
            print(f" it {it}: no solution ({st})", flush=True)
            continue
        xs = np.array(h2.getSolution().col_value)
        md = {}
        for i in modes:
            if i in mdl.fv and xs[mdl.fv[i]] > 0.5: md[i] = "full"
            elif i in mdl.sv and xs[mdl.sv[i]] > 0.5: md[i] = "src"
            elif i in mdl.hv and xs[mdl.hv[i]] > 0.5:
                md[i] = "half" + "".join(t for t in M[i]["dataRequirements"] if xs[mdl.zv[(i, t)]] > 0.5)
        v = val(md)
        print(f" it {it}: |N| {len(N)} obj {info.objective_function_value:.0f} bound {info.mip_dual_bound:.1f} "
              f"val {v} best {best} ({time.time() - t0:.0f}s)", flush=True)
        if v > best:
            best = v
            inc = md
            plan = {"full": [i for i, m in md.items() if m == "full"],
                    "half": [i for i, m in md.items() if m.startswith("half")],
                    "source": [i for i, m in md.items() if m == "src"],
                    "srcreq": [[i, t] for i, m in md.items() if m.startswith("half") for t in m[4:]],
                    "obj": v}
            json.dump(plan, open(out, "w"))


if __name__ == "__main__":
    main()
