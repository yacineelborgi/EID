"""Restricted MIP on the rigorous relaxation: LP -> exact mode deviations -> fix models whose
second-best mode is more than D away (heuristic unless D >= U - target) -> MIP.

usage: python rmip5.py taus D time_limit margin_free margin_tot out_plan [start_plan]
"""
import json
import sys
import time

import numpy as np
import highspy

from relax import INF, TYPES, build, to_highs
from modedev import mode_devs

inst = json.load(open("/home/user/EID/datasets/5_energy.json"))
M = inst["models"]


def set_mode(mdl, lo, up, i, mode):
    vs = []
    if i in mdl.fv: vs.append((mdl.fv[i], mode == "full"))
    if i in mdl.sv: vs.append((mdl.sv[i], mode == "src"))
    if i in mdl.hv:
        vs.append((mdl.hv[i], mode.startswith("half")))
        for t in M[i]["dataRequirements"]:
            vs.append((mdl.zv[(i, t)], mode.startswith("half") and t in mode[4:]))
    for j, v in vs:
        lo[j] = up[j] = 1 if v else 0


def plan_mode(plan):
    md = {}
    srq = {}
    for i, t in plan["srcreq"]:
        srq.setdefault(i, "")
        srq[i] += t
    for i in plan["full"]: md[i] = "full"
    for i in plan["source"]: md[i] = "src"
    for i in plan["half"]: md[i] = "half" + "".join(t for t in M[i]["dataRequirements"] if t in srq.get(i, ""))
    return md


def run(taus, D, tl, mf, mt, start=None, prune=40, verbose=False, gap=0.0, fixed_modes=None):
    t0 = time.time()
    mdl = build(inst, taus)
    for r, nm in enumerate(mdl.rnames):
        if nm.startswith("free"): mdl.rhi[r] -= mf
        if nm.startswith("tot"): mdl.rhi[r] -= mt
    h, _ = to_highs(mdl, integer=False)
    h.run()
    sol = h.getSolution()
    y = np.array(sol.row_dual)
    x = np.array(sol.col_value)
    U, modes, red = mode_devs(mdl, y)
    if U > 1e9:
        U, modes, red = mode_devs(mdl, -y)
    print(f"LP {h.getInfo().objective_function_value:.3f} U {float(U):.3f} ({time.time() - t0:.0f}s)", flush=True)
    lo, up = list(mdl.lo), list(mdl.up)
    nfree = 0
    for i, d in modes.items():
        vals = sorted(d.values())
        if len(vals) > 1 and float(vals[1]) <= D:
            nfree += 1
            continue
        best = min(d, key=d.get)
        set_mode(mdl, lo, up, i, best)
    if fixed_modes:
        for i, md in fixed_modes.items():
            set_mode(mdl, lo, up, i, md)
    print("free models", nfree, flush=True)
    # prune tangent rows around LP active piece
    act = np.array(sol.row_value)
    keep = np.ones(len(mdl.rlo), bool)
    groups = {}
    for r, nm in enumerate(mdl.rnames):
        if (nm.startswith("tF") or nm.startswith("t")) and "_" in nm and not nm.startswith("tot"):
            g, j = nm.rsplit("_", 1)
            groups.setdefault(g, []).append((int(j), r))
    for g, lst in groups.items():
        sl = [(act[r] - mdl.rlo[r], j) for j, r in lst]
        jb = min(sl)[1]
        for j, r in lst:
            if abs(j - jb) > prune:
                keep[r] = False
    rm = -np.ones(len(mdl.rlo), int); rm[keep] = np.arange(keep.sum())
    m2 = type(mdl)()
    m2.c, m2.lo, m2.up, m2.integ, m2.names = mdl.c, lo, up, mdl.integ, mdl.names
    R = np.array(mdl.rows); sel = keep[R]
    m2.rows = list(rm[R[sel]]); m2.cols = list(np.array(mdl.cols)[sel]); m2.vals = list(np.array(mdl.vals)[sel])
    m2.rlo = [v for v, k in zip(mdl.rlo, keep) if k]; m2.rhi = [v for v, k in zip(mdl.rhi, keep) if k]
    m2.rnames = [v for v, k in zip(mdl.rnames, keep) if k]
    h2, _ = to_highs(m2, integer=True)
    h2.setOptionValue("time_limit", float(tl))
    h2.setOptionValue("mip_rel_gap", gap)
    h2.setOptionValue("mip_abs_gap", 0.99)
    h2.setOptionValue("output_flag", verbose)
    h2.setOptionValue("threads", 1)
    if start is not None:
        sm = plan_mode(start)
        xs = np.zeros(len(mdl.c))
        for i, md in sm.items():
            l2, u2 = [0] * len(mdl.c), [0] * len(mdl.c)
            set_mode(mdl, l2, u2, i, md)
            for j in range(len(mdl.c)):
                pass
        # simpler: build explicit vector
        xs = np.zeros(len(mdl.c))
        for i, md in sm.items():
            if md == "full": xs[mdl.fv[i]] = 1
            elif md == "src": xs[mdl.sv[i]] = 1
            else:
                xs[mdl.hv[i]] = 1
                for t in md[4:]: xs[mdl.zv[(i, t)]] = 1
        hs = highspy.HighsSolution()
        hs.col_value = list(xs)
        hs.value_valid = True
        print("setSolution", h2.setSolution(hs), flush=True)
    h2.run()
    info = h2.getInfo()
    xs = np.array(h2.getSolution().col_value)
    print(f"MIP obj {info.objective_function_value:.1f} bound {info.mip_dual_bound:.2f} ({time.time() - t0:.0f}s)", flush=True)
    plan = {"full": [], "half": [], "source": [], "srcreq": []}
    for i, j in mdl.fv.items():
        if xs[j] > 0.5: plan["full"].append(i)
    for i, j in mdl.hv.items():
        if xs[j] > 0.5: plan["half"].append(i)
    for i, j in mdl.sv.items():
        if xs[j] > 0.5: plan["source"].append(i)
    for (i, t), j in mdl.zv.items():
        if xs[j] > 0.5: plan["srcreq"].append([i, t])
    plan["obj"] = info.objective_function_value
    plan["bound"] = info.mip_dual_bound
    plan["U"] = float(U)
    rv = np.array(h2.getSolution().row_value)
    plan["slack"] = {nm: m2.rhi[r] - rv[r] for r, nm in enumerate(m2.rnames)
                     if nm.startswith("free") or nm.startswith("tot")}
    return plan


if __name__ == "__main__":
    a = sys.argv
    taus = tuple(int(x) for x in a[1].split(","))
    start = json.load(open(a[7])) if len(a) > 7 else None
    plan = run(taus, float(a[2]), float(a[3]), int(a[4]), int(a[5]), start=start)
    json.dump(plan, open(a[6], "w"))
    print({k: round(v) for k, v in plan["slack"].items() if k[-2:] in ("n1", "t1", "i1", "c1")})
