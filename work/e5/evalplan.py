"""Pack a plan with exact_assign, finalize, greedy add; keep the best legit solution.
usage: python evalplan.py plan.json out.json [orders] [seeds]"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/user/EID")
from exact_assign import TYPES, assign_exact
from solve import finalize, greedy_add, score_of, strict_ok

INST = "/home/user/EID/datasets/5_energy.json"
txt = open(INST).read()
inst = json.loads(txt)
M, D = inst["models"], inst["datasets"]


def evaluate(plan, out, orders=("lb_asc", "width"), seeds=(0,), budgets=(None,), verbose=True):
    best = -1
    if os.path.exists(out):
        best, ok, _ = score_of(txt, json.load(open(out)))
    res = []
    for order in orders:
        for seed in (seeds if order == "random" else (0,)):
            for bf in budgets:
                t0 = time.time()
                dm, failed, st = assign_exact(inst, plan, seed=seed, order=order, budget_frac=bf)
                dm2, mm, full, half, us = finalize(inst, plan, dm, failed)
                pre = sum(M[i]["value"] for i in full) + sum(M[i]["value"] // 2 for i in half)
                added = greedy_add(inst, dm2, mm, full, half, us)
                sol = {"dataMappings": [[d, m] for d, m in dm2.items()], "modelMappings": mm}
                sc, ok, msg = score_of(txt, sol)
                ok = ok and strict_ok(inst, sol)
                if verbose:
                    print(f"   {order} s{seed} b{bf}: failed {sorted(failed)} pre {pre} added {added} score {sc} {ok} {msg} "
                          f"{time.time() - t0:.0f}s", flush=True)
                    print("     ", {t: {k: st[t][k] for k in ("slackA", "slackT", "wA", "wB", "fA", "fB", "left")} for t in TYPES})
                res.append((sc, ok, failed, st))
                if ok and sc > best:
                    best = sc
                    json.dump(sol, open(out, "w"))
                    print("   wrote", out, sc, flush=True)
    return best, res


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    orders = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("lb_asc", "width")
    seeds = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (0,)
    print("plan obj", plan.get("obj"), flush=True)
    evaluate(plan, sys.argv[2], orders, seeds)
