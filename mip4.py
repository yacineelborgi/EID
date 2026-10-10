"""Exact-ish plan MIP for instances without mono-requirement models (no sources).

max sum v f + (v//2) h
 s.t. f_i + h_i <= 1
      sum_i e_i (f_i + h_i) <= cap
      sum_i lb_it f_i <= F_t - sf_t            (free data)
      sum_i lb_it (f_i + h_i) <= T_t - st_t    (all data)

usage: python mip4.py datasets/4_precise.json time_limit margin_free [out_plan] [core_size]
"""
import json
import sys
import time

import highspy
import numpy as np

TYPES = "ntic"


def build(inst, margin_free, margin_tot=0, lp=False, fix=None, hint=None):
    M, D = inst["models"], inst["datasets"]
    cap = inst["energyCap"]
    F = {t: 0 for t in TYPES}
    T = {t: 0 for t in TYPES}
    for d in D:
        T[d["dataType"]] += d["size"]
        if not d["isCopyrighted"]:
            F[d["dataType"]] += d["size"]
    if isinstance(margin_free, (int, float)):
        margin_free = {t: margin_free for t in TYPES}
    ok = [i for i, m in enumerate(M)
          if m["value"] > 0 and all(r["upperBound"] >= r["lowerBound"] for r in m["dataRequirements"].values())
          and all(len(m["dataRequirements"]) > 1 for _ in [0])]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    n = len(ok)
    inf = highspy.kHighsInf
    # columns: f_0..f_{n-1}, h_0..h_{n-1}
    cost = []
    for i in ok:
        cost.append(-M[i]["value"])
    for i in ok:
        cost.append(-(M[i]["value"] // 2))
    lower = [0.0] * (2 * n)
    upper = [1.0] * (2 * n)
    if fix:
        for k, i in enumerate(ok):
            if i in fix:
                lower[k] = upper[k] = float(fix[i][0])
                lower[n + k] = upper[n + k] = float(fix[i][1])
    h.addVars(2 * n, np.array(lower), np.array(upper))
    h.changeColsCost(2 * n, np.arange(2 * n, dtype=np.int32), np.array(cost, float))
    if not lp:
        h.changeColsIntegrality(2 * n, np.arange(2 * n, dtype=np.int32),
                                np.array([highspy.HighsVarType.kInteger] * (2 * n)))
    # f+h<=1
    for k in range(n):
        h.addRow(-inf, 1.0, 2, np.array([k, n + k], np.int32), np.array([1.0, 1.0]))
    # energy
    h.addRow(-inf, 1.0 - 1e-7, 2 * n, np.arange(2 * n, dtype=np.int32),
             np.array([M[i]["energyCost"] / cap for i in ok] * 2))
    for t in TYPES:
        idx, val = [], []
        for k, i in enumerate(ok):
            r = M[i]["dataRequirements"].get(t)
            if r:
                idx.append(k); val.append(float(r["lowerBound"]))
        h.addRow(-inf, float(F[t] - margin_free[t]), len(idx), np.array(idx, np.int32), np.array(val))
        h.addRow(-inf, float(T[t] - margin_tot), 2 * len(idx),
                 np.array(idx + [n + k for k in idx], np.int32), np.array(val + val))
    if hint is not None:
        sol = highspy.HighsSolution()
        x = [0.0] * (2 * n)
        for k, i in enumerate(ok):
            if i in hint["full"]:
                x[k] = 1.0
            if i in hint["half"]:
                x[n + k] = 1.0
        sol.col_value = x
        h.setSolution(sol)
    return h, ok


def plan_from(h, ok):
    x = h.getSolution().col_value
    n = len(ok)
    full = [ok[k] for k in range(n) if x[k] > 0.5]
    half = [ok[k] for k in range(n) if x[n + k] > 0.5]
    return full, half


def lp_info(inst, margin):
    h, ok = build(inst, margin, lp=True)
    h.run()
    info = h.getInfo()
    sol = h.getSolution()
    duals = sol.row_dual
    n = len(ok)
    return -info.objective_function_value, sol.col_value, duals, ok


def core_solve(inst, margin, tl, core_frac_list=(300, 600, 1200), hint=None):
    """Solve LP, fix variables with large reduced cost, MIP on the core."""
    M = inst["models"]
    lpv, x, duals, ok = lp_info(inst, margin)
    n = len(ok)
    # rows: n rows of f+h<=1, then energy, then for each type: free, tot
    lamE = duals[n]
    mu = {}
    nu = {}
    for j, t in enumerate(TYPES):
        mu[t] = duals[n + 1 + 2 * j]
        nu[t] = duals[n + 2 + 2 * j]
    print("LP bound", lpv, "duals E", lamE, "free", mu, "tot", nu, flush=True)
    # reduced profit of each mode (duals are <= 0 for <= rows in a min problem)
    rc = []
    for k, i in enumerate(ok):
        m = M[i]
        pf = m["value"] + lamE * m["energyCost"] / inst["energyCap"] + sum((mu[t] + nu[t]) * r["lowerBound"] for t, r in m["dataRequirements"].items())
        ph = m["value"] // 2 + lamE * m["energyCost"] / inst["energyCap"] + sum(nu[t] * r["lowerBound"] for t, r in m["dataRequirements"].items())
        rc.append((pf, ph))
    return lpv, rc, ok, x


if __name__ == "__main__":
    path = sys.argv[1]
    tl = float(sys.argv[2])
    margin = float(sys.argv[3])
    out = sys.argv[4] if len(sys.argv) > 4 else None
    core = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    inst = json.load(open(path))
    M = inst["models"]
    t0 = time.time()
    hint = None
    if len(sys.argv) > 6:
        hint = json.load(open(sys.argv[6]))
        hint = {"full": set(hint["full"]), "half": set(hint["half"])}
    fix = None
    if core:
        lpv, rc, ok, x = core_solve(inst, margin, tl)
        # score: distance of each model from being indifferent
        sc = []
        for k, i in enumerate(ok):
            pf, ph = rc[k]
            best = max(pf, ph, 0.0)
            # how close is the second best option to the best one
            opts = sorted([pf, ph, 0.0], reverse=True)
            sc.append((opts[0] - opts[1], i, pf, ph))
        sc.sort()
        free_set = {i for _, i, _, _ in sc[:core]}
        fix = {}
        for d, i, pf, ph in sc[core:]:
            if pf >= ph and pf > 0:
                fix[i] = (1, 0)
            elif ph > pf and ph > 0:
                fix[i] = (0, 1)
            else:
                fix[i] = (0, 0)
        print("core", core, "max gap in core", sc[core - 1][0], "fixed", len(fix), flush=True)
    h, ok = build(inst, margin, fix=fix, hint=hint)
    h.setOptionValue("time_limit", tl)
    h.setOptionValue("mip_rel_gap", 1e-9)
    h.setOptionValue("mip_abs_gap", 0.5)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", True)
    h.run()
    info = h.getInfo()
    full, half = plan_from(h, ok)
    obj = sum(M[i]["value"] for i in full) + sum(M[i]["value"] // 2 for i in half)
    print("obj", obj, "bound", -info.mip_dual_bound, "full", len(full), "half", len(half),
          "time", time.time() - t0, flush=True)
    if out:
        json.dump({"full": full, "half": half, "source": [], "srcreq": [], "obj": obj,
                   "bound": -info.mip_dual_bound, "margin": margin}, open(out, "w"))
