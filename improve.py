"""Residual rounds: keep the current best solution, re-solve the MIP on what is left
(unused models, datasets and energy) and try to add more models.

usage: python improve.py datasets/6_shortage.json [rounds] [mip_time] [alpha]
"""
import json
import os
import sys

from assign import TYPES, assign
from select_mip import build_and_solve
from solve import finalize, greedy_add, save_if_better, score_of


def main(path, rounds=3, tl=120, alpha=0.97):
    with open(path) as f:
        txt = f.read()
    inst = json.loads(txt)
    M, D = inst["models"], inst["datasets"]
    out = os.path.join("solutions", os.path.basename(path).replace(".json", "_submission.json"))
    with open(out) as f:
        sol = json.load(f)
    best, ok, _ = score_of(txt, sol)
    assert ok
    print("start", best, flush=True)
    for rnd in range(rounds):
        dm = {d: m for d, m in sol["dataMappings"]}
        mm = sol["modelMappings"]
        used = set(dm.values()) | {x for p in mm for x in p}
        energy = sum(M[i]["energyCost"] for i in used)
        F = {t: 0 for t in TYPES}
        T = {t: 0 for t in TYPES}
        for d in D:
            if d["id"] not in dm:
                T[d["dataType"]] += d["size"]
                if not d["isCopyrighted"]:
                    F[d["dataType"]] += d["size"]
        resid = {"F": F, "T": T, "cap": inst["energyCap"] - energy, "spare": {t: 0 for t in TYPES}}
        plan = build_and_solve(inst, alpha, alpha, time_limit=tl, forced_off=sorted(used), resid=resid)
        print(f" round {rnd}: resid MIP obj {plan['obj']:.0f} full {len(plan['full'])} "
              f"half {len(plan['half'])} src {len(plan['source'])}", flush=True)
        cand = None
        for order in ("lb_asc", "width", "lb_desc"):
            ndm, failed = assign(inst, plan, order=order, exclude=dm.keys())
            ndm, nmm, full, half, us = finalize(inst, plan, ndm, failed)
            mdm = dict(dm); mdm.update(ndm)
            mmm = mm + nmm
            # greedy_add needs the sets of all used models
            src_old = {s for s, _ in mm}
            tgt = set(dm.values()) - src_old
            fullA = full | {i for i in tgt if all(not D[d]["isCopyrighted"] for d, m in dm.items() if m == i)
                            and i not in {t for _, t in mm}}
            halfA = half | (tgt - fullA)
            greedy_add(inst, mdm, mmm, fullA, halfA, us | src_old)
            s2 = {"dataMappings": [[d, m] for d, m in mdm.items()], "modelMappings": mmm}
            sc, ok2, msg = score_of(txt, s2)
            print(f"   {order}: failed {len(failed)} score {sc} {'OK' if ok2 else msg}", flush=True)
            if ok2 and sc > (cand[0] if cand else best):
                cand = (sc, s2)
        if not cand:
            alpha *= 0.98
            continue
        best, sol = cand
        save_if_better(path, best, sol)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 3, float(a[3]) if len(a) > 3 else 120,
         float(a[4]) if len(a) > 4 else 0.97)
