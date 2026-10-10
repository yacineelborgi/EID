"""Evaluate a cached plan with the exact assigner (several orders / options), keep the best.

usage: python eval_plan.py datasets/X.json plan.json out.json orders [G0s]
"""
import json
import os
import sys
import time

from exact_assign import assign_exact
from solve import finalize, greedy_add, score_of, strict_ok

path, pf, out = sys.argv[1], sys.argv[2], sys.argv[3]
orders = sys.argv[4].split(",")
G0s = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [20000]
txt = open(path).read()
inst = json.loads(txt)
M = inst["models"]
plan = json.load(open(pf))
best = -1
if os.path.exists(out):
    best, ok, _ = score_of(txt, json.load(open(out)))
    best = best if ok else -1
print("start", best, "plan obj", plan.get("obj"), flush=True)
for order in orders:
    for G0 in G0s:
        for demote in (True, False):
            for bf in (None, 1.0):
                t0 = time.time()
                dm, failed, st = assign_exact(inst, plan, order=order, G0=G0, budget_frac=bf, demote=demote)
                dm2, mm, full, half, us = finalize(inst, plan, dm, failed)
                added = greedy_add(inst, dm2, mm, full, half, us)
                sol = {"dataMappings": [[d, m] for d, m in dm2.items()], "modelMappings": mm}
                sc, ok, msg = score_of(txt, sol)
                ok = ok and strict_ok(inst, sol)
                print(f" {order} G0 {G0} dem {demote} b{bf}: failed {len(failed)} demoted {st['demoted']} "
                      f"added {added} score {sc} {ok} {msg} {time.time() - t0:.0f}s", flush=True)
                if ok and sc > best:
                    best = sc
                    json.dump(sol, open(out, "w"))
                    print("  wrote", out, sc, flush=True)
