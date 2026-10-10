"""Waste-corrected LP -> reduced-cost fixing -> restricted MIP (warm start) -> exact assignment.

usage: python big_opt.py datasets/7_big.json start_solution.json out.json thr mip_time margin [extra_waste_iters]
"""
import json
import os
import sys
import time

import numpy as np

import rmip
from exact_assign import TYPES, assign_exact
from solve import finalize, greedy_add, score_of, strict_ok

path, start_sol, out = sys.argv[1], sys.argv[2], sys.argv[3]
thr = float(sys.argv[4]); tl = float(sys.argv[5]); margin = int(sys.argv[6])
iters = int(sys.argv[7]) if len(sys.argv) > 7 else 1
orders = sys.argv[8].split(",") if len(sys.argv) > 8 else ["lb_asc", "width"]
txt = open(path).read()
inst = json.loads(txt)
M, D = inst["models"], inst["datasets"]
cap = inst["energyCap"]
F = {t: 0 for t in TYPES}
T = {t: 0 for t in TYPES}
for d in D:
    T[d["dataType"]] += d["size"]
    if not d["isCopyrighted"]:
        F[d["dataType"]] += d["size"]
sol0 = json.load(open(start_sol))
best, ok, _ = score_of(txt, sol0)
if os.path.exists(out):
    b2, ok2, _ = score_of(txt, json.load(open(out)))
    if ok2:
        best = max(best, b2)
print("start", best, flush=True)

mod0 = rmip.build(inst, F, T, cap)
x0 = rmip.from_solution(inst, mod0, sol0)
plan = rmip.to_plan(mod0, x0)
print("start plan obj", float(-(mod0["c"] @ x0)), flush=True)
_, _, st = assign_exact(inst, plan, order="lb_asc", budget_frac=None)
wA = {t: st[t]["wA"] for t in TYPES}
wB = {t: st[t]["wB"] for t in TYPES}


def evaluate(plan):
    global best
    for order in orders:
        for bf in (None, 1.0):
            t0 = time.time()
            dm, failed, st = assign_exact(inst, plan, order=order, budget_frac=bf)
            dm2, mm, full, half, us = finalize(inst, plan, dm, failed)
            pre = sum(M[i]["value"] for i in full) + sum(M[i]["value"] // 2 for i in half)
            added = greedy_add(inst, dm2, mm, full, half, us)
            sol = {"dataMappings": [[d, m] for d, m in dm2.items()], "modelMappings": mm}
            sc, ok, msg = score_of(txt, sol)
            ok = ok and strict_ok(inst, sol)
            print(f"   {order} b{bf}: failed {len(failed)} pre {pre} added {added} score {sc} {ok} {msg} "
                  f"{time.time() - t0:.0f}s", flush=True)
            if ok and sc > best:
                best = sc
                json.dump(sol, open(out, "w"))
                print("   wrote", out, sc, flush=True)
            if order == "lb_asc" and bf is None:
                yield st


for it in range(iters):
    print("iter", it, "waste A", wA, "B", wB, flush=True)
    Fc = {t: F[t] - wA[t] - margin for t in TYPES}
    Tc = {t: T[t] - wA[t] - wB[t] - margin for t in TYPES}
    mod = rmip.build(inst, Fc, Tc, cap)
    t0 = time.time()
    lp = rmip.run(mod, time_limit=600)
    x, rc = lp["x"], lp["rc"]
    print(f" LP obj {lp['obj']:.1f} {lp['status']} {time.time() - t0:.0f}s", flush=True)
    lbv, ubv = mod["lbv"].copy(), mod["ubv"].copy()
    at0 = (x < 1e-7) & (rc > thr)
    at1 = (x > 1 - 1e-7) & (rc < -thr)
    ubv[at0] = 0
    lbv[at1] = 1
    nM = mod["nM"]
    freev = (~at0) & (~at1) & (mod["ubv"] > 0)
    fm = set(np.nonzero(freev[:3 * nM])[0] % nM)
    print(f" fixed0 {at0.sum()} fixed1 {at1.sum()} free vars {freev.sum()} free models {len(fm)}", flush=True)
    xs = rmip.from_solution(inst, mod, sol0)
    t0 = time.time()
    r = rmip.run(mod, lbv=lbv, ubv=ubv, integer=True, time_limit=tl, start=None if os.environ.get("NOSTART") else xs)
    print(f" MIP obj {r['obj']:.1f} bound {r['bound']:.1f} {r['status']} {time.time() - t0:.0f}s", flush=True)
    if not np.isfinite(r["obj"]) or len(r["x"]) != mod["nV"]:
        break
    plan = rmip.to_plan(mod, r["x"])
    json.dump(plan, open(f"plans/{os.path.basename(path)[:-5]}_rmip_{it}.json", "w"))
    stn = None
    for s in evaluate(plan):
        stn = s
    if stn is not None:
        wA = {t: max(wA[t], stn[t]["wA"]) for t in TYPES}
        wB = {t: stn[t]["wB"] for t in TYPES}
