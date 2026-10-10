"""LP -> safe bound -> prune tangent rows -> MIP. Writes plan json (full/half/source/srcreq).

usage: python mip.py inst taus time_limit out_plan [extra_json]
extra_json: {"forbid_full": [...], "forbid_half": [...], "rowcap": {"free_n_1": -500, ...}}
"""
import json
import sys
import time

import numpy as np

from relax import INF, TYPES, build, safe_bound, to_highs
import highspy


def solve(inst, taus, tl, extra=None, verbose=False, start=None, gap=0.0, prune=60):
    t0 = time.time()
    mdl = build(inst, taus)
    extra = extra or {}
    lo = list(mdl.lo); up = list(mdl.up)
    for i in extra.get("forbid_full", []):
        if i in mdl.fv: up[mdl.fv[i]] = 0
    for i in extra.get("forbid_half", []):
        if i in mdl.hv: up[mdl.hv[i]] = 0
    for i in extra.get("force_full", []):
        lo[mdl.fv[i]] = 1
    h, A = to_highs(mdl, integer=False, lo=lo, up=up)
    h.run()
    lpobj = h.getInfo().objective_function_value
    sol = h.getSolution()
    y = np.array(sol.row_dual)
    B, red = safe_bound(mdl, y, lo, up)
    if B is None:
        B, red = safe_bound(mdl, -y, lo, up)
    print(f"LP {lpobj:.4f} safe {float(B):.4f} ({time.time() - t0:.0f}s)", flush=True)
    x = np.array(sol.col_value)
    # prune tangent rows: keep tangent rows whose j is within +-prune of active k
    keep = np.ones(len(mdl.rlo), bool)
    act = np.array(sol.row_value)
    if prune:
        groups = {}
        for r, nm in enumerate(mdl.rnames):
            if nm.startswith("tF") or (nm.startswith("t") and "_" in nm and nm[1] in TYPES):
                g, j = nm.rsplit("_", 1)
                groups.setdefault(g, []).append((int(j), r))
        for g, lst in groups.items():
            # active = row with smallest slack
            sl = [(act[r] - mdl.rlo[r], j) for j, r in lst]
            jbest = min(sl)[1]
            for j, r in lst:
                if abs(j - jbest) > prune:
                    keep[r] = False
    rows_map = -np.ones(len(mdl.rlo), int)
    rows_map[keep] = np.arange(keep.sum())
    m2 = type(mdl)()
    m2.c, m2.lo, m2.up, m2.integ, m2.names = mdl.c, lo, up, mdl.integ, mdl.names
    sel = keep[np.array(mdl.rows)]
    m2.rows = list(rows_map[np.array(mdl.rows)[sel]])
    m2.cols = list(np.array(mdl.cols)[sel]); m2.vals = list(np.array(mdl.vals)[sel])
    m2.rlo = [v for v, k in zip(mdl.rlo, keep) if k]; m2.rhi = [v for v, k in zip(mdl.rhi, keep) if k]
    m2.rnames = [v for v, k in zip(mdl.rnames, keep) if k]
    h2, _ = to_highs(m2, integer=True)
    h2.setOptionValue("time_limit", float(tl))
    h2.setOptionValue("mip_rel_gap", gap)
    h2.setOptionValue("mip_abs_gap", 0.99)
    h2.setOptionValue("output_flag", verbose)
    if start is not None:
        hs = highspy.HighsSolution()
        hs.col_value = list(start)
        hs.value_valid = True
        h2.setSolution(hs)
    h2.run()
    info = h2.getInfo()
    xs = np.array(h2.getSolution().col_value)
    print(f"MIP obj {info.objective_function_value:.1f} bound {info.mip_dual_bound:.2f} "
          f"gap {info.mip_gap:.2e} ({time.time() - t0:.0f}s) rows {len(m2.rlo)}", flush=True)
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
    plan["lp"] = lpobj
    plan["safe"] = float(B)
    # slacks of data rows
    rv = np.array(h2.getSolution().row_value)
    sl = {}
    for r, nm in enumerate(m2.rnames):
        if nm.startswith("free") or nm.startswith("tot"):
            sl[nm] = m2.rhi[r] - rv[r]
    plan["slack"] = sl
    return plan, mdl, m2, xs


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    taus = tuple(int(x) for x in sys.argv[2].split(","))
    extra = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
    plan, mdl, m2, xs = solve(inst, taus, float(sys.argv[3]), extra, verbose=False)
    json.dump(plan, open(sys.argv[4], "w"))
    print({k: round(v) for k, v in plan["slack"].items()})
