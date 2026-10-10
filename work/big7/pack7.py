"""Pack a plan with exact_assign (several orders), finalize, greedy fill, check, save if better.
usage: pack7.py plan.json [orders] [G0s]"""
import sys, json, time, os
sys.path.insert(0, "/home/user/EID")
from exact_assign import assign_exact
from solve import finalize, greedy_add, score_of, strict_ok
OUT = "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_max/7_big_submission.json"
LOCAL = "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7/best7.json"
CUR = 5822389
txt = open("/home/user/EID/datasets/7_big.json").read()
inst = json.loads(txt)
M = inst["models"]
plan = json.load(open(sys.argv[1]))
orders = sys.argv[2].split(",") if len(sys.argv) > 2 else ["lb_asc"]
G0s = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [20000]
best = CUR
for f in (OUT, LOCAL):
    if os.path.exists(f):
        s, ok, _ = score_of(txt, json.load(open(f)))
        if ok: best = max(best, s)
pv = sum(M[i]["value"] for i in plan["full"]) + sum(M[i]["value"] // 2 for i in plan["half"])
print("best so far", best, "plan value", pv, flush=True)
for order in orders:
    for G0 in G0s:
        for bf in (None,):
            t0 = time.time()
            dm, failed, st = assign_exact(inst, plan, order=order, G0=G0, budget_frac=bf)
            fl = {i: M[i]["value"] if i in set(plan["full"]) else M[i]["value"] // 2 for i in failed}
            dm2, mm, full, half, us = finalize(inst, plan, dm, failed)
            pre = sum(M[i]["value"] for i in full) + sum(M[i]["value"] // 2 for i in half)
            added = greedy_add(inst, dm2, mm, full, half, us)
            sol = {"dataMappings": [[d, m] for d, m in dm2.items()], "modelMappings": mm}
            sc, ok, msg = score_of(txt, sol)
            ok = ok and strict_ok(inst, sol)
            print(f" {order} G0 {G0}: failed {len(failed)} {fl} pre {pre} added {added} score {sc} {ok} {msg} {time.time()-t0:.0f}s", flush=True)
            print("   stats", {t: st[t] for t in "ntic"}, flush=True)
            if ok and sc > best:
                best = sc
                json.dump(sol, open(LOCAL, "w"))
                json.dump(sol, open(OUT, "w"))
                print("  wrote", sc, flush=True)
