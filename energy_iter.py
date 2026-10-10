"""Waste-corrected MIP: measure the unavoidable waste of an exact assignment (tiny bins
need whole datasets), remove it from the per-type capacities, re-solve the MIP and assign.

usage: python energy_iter.py datasets/5_energy.json start_plan.json tl margin out.json tag
"""
import json
import os
import sys
import time

from exact_assign import TYPES, assign_exact
from select_mip import build_and_solve
from solve import finalize, greedy_add, score_of, strict_ok

path, pf0, tl, margin, out, tag = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
txt = open(path).read()
inst = json.loads(txt)
M, D = inst["models"], inst["datasets"]
F = {t: 0 for t in TYPES}
T = {t: 0 for t in TYPES}
for d in D:
    T[d["dataType"]] += d["size"]
    if not d["isCopyrighted"]:
        F[d["dataType"]] += d["size"]
best = -1
if os.path.exists(out):
    best, ok, _ = score_of(txt, json.load(open(out)))
print("start best", best, flush=True)


def evaluate(plan, orders=("lb_asc", "width", "random"), seeds=(0, 1), budgets=(1.0, None)):
    global best
    res_stats = None
    for order in orders:
        for seed in (seeds if order == "random" else (0,)):
            for bf in budgets:
                t0 = time.time()
                dm, failed, st = assign_exact(inst, plan, seed=seed, order=order, budget_frac=bf)
                if res_stats is None and order == "lb_asc" and bf is None:
                    res_stats = st
                dm2, mm, full, half, us = finalize(inst, plan, dm, failed)
                pre = sum(M[i]["value"] for i in full) + sum(M[i]["value"] // 2 for i in half)
                added = greedy_add(inst, dm2, mm, full, half, us)
                sol = {"dataMappings": [[d, m] for d, m in dm2.items()], "modelMappings": mm}
                sc, ok, msg = score_of(txt, sol)
                ok = ok and strict_ok(inst, sol)
                print(f"   {order} s{seed} b{bf}: failed {len(failed)} pre {pre} added {added} score {sc} {ok} {msg} "
                      f"{time.time() - t0:.0f}s", flush=True)
                if ok and sc > best:
                    best = sc
                    json.dump(sol, open(out, "w"))
                    print("   wrote", out, sc, flush=True)
    return res_stats


plan = json.load(open(pf0))
print("plan0 obj", plan["obj"], flush=True)
_, _, st = assign_exact(inst, plan, order="lb_asc", budget_frac=None)
for it in range(3):
    wA = {t: st[t]["wA"] for t in TYPES}
    wB = {t: st[t]["wB"] for t in TYPES}
    print("iter", it, "waste A", wA, "waste B", wB, flush=True)
    resid = {"F": {t: F[t] - wA[t] - margin for t in TYPES},
             "T": {t: T[t] - wA[t] - wB[t] - margin for t in TYPES},
             "cap": inst["energyCap"], "spare": {t: 0 for t in TYPES}}
    t0 = time.time()
    plan = build_and_solve(inst, 1.0, 1.0, time_limit=tl, resid=resid)
    json.dump(plan, open(f"plans/5_energy_corr_{tag}_{it}.json", "w"))
    print(f"iter {it} MIP obj {plan['obj']:.0f} bound {plan['bound']:.0f} {time.time() - t0:.0f}s", flush=True)
    st2 = evaluate(plan)
    if st2 is not None:
        st = st2
