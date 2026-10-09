"""Full pipeline: MIP selection -> dataset assignment -> repair/greedy fill -> save.

usage: python solve.py datasets/2_medium.json [mip_time_limit] [alpha]
"""
import json
import os
import sys

from assign import TYPES, Pool, assign, fill_bin
from select_mip import build_and_solve
from test_solution import getSolutionScore


def finalize(inst, plan, data_map, failed):
    M = inst["models"]
    full = set(plan["full"]) - failed
    half = set(plan["half"]) - failed
    sources = [i for i in plan["source"] if i not in failed]
    # match sources to sourced requirements
    by_type = {t: [] for t in TYPES}
    for i in sources:
        by_type[next(iter(M[i]["dataRequirements"]))].append(i)
    model_map = []
    reqs = {t: [] for t in TYPES}
    for i, t in plan["srcreq"]:
        if i in half:
            reqs[t].append(i)
    unmatched = set()
    for t in TYPES:
        # highest value targets first
        reqs[t].sort(key=lambda i: -M[i]["value"])
        for k, i in enumerate(reqs[t]):
            if k < len(by_type[t]):
                model_map.append([by_type[t][k], i])
            else:
                unmatched.add(i)
    if unmatched:
        half -= unmatched
        model_map = [mm for mm in model_map if mm[1] in half]
    used_src = {s for s, _ in model_map}
    keep = full | half | used_src
    data_map = {d: m for d, m in data_map.items() if m in keep}
    return data_map, model_map, full, half, used_src


def greedy_add(inst, data_map, model_map, full, half, used_src):
    """Try to add unused models using leftover energy and datasets."""
    M = inst["models"]
    D = inst["datasets"]
    cap = inst["energyCap"]
    energy = sum(M[i]["energyCost"] for i in full | half | used_src)
    free = {t: Pool([]) for t in TYPES}
    copy = {t: Pool([]) for t in TYPES}
    for d in D:
        if d["id"] not in data_map:
            (copy if d["isCopyrighted"] else free)[d["dataType"]].put((d["size"], d["id"]))
    used = full | half | used_src
    cand = [i for i in range(len(M)) if i not in used and M[i]["value"] > 0]
    cand.sort(key=lambda i: -M[i]["value"] / max(M[i]["energyCost"], 1))
    added = 0
    for i in cand:
        m = M[i]
        if energy + m["energyCost"] > cap:
            continue
        reqs = m["dataRequirements"]
        if any(r["upperBound"] < r["lowerBound"] for r in reqs.values()):
            continue
        for mode in ("full", "half"):
            taken = []
            ok = True
            for t, r in reqs.items():
                pools = [free[t]] if mode == "full" else [copy[t], free[t]]
                res = fill_bin(r["lowerBound"], r["upperBound"], pools)
                if res is None:
                    ok = False
                    break
                taken += res
            if ok:
                for _, it in taken:
                    data_map[it[1]] = i
                (full if mode == "full" else half).add(i)
                energy += m["energyCost"]
                added += 1
                break
            for p, it in taken:
                p.put(it)
    return added


def score_of(inst_txt, sol):
    return getSolutionScore(json.dumps(sol), inst_txt, show_details=False)


def run(path, tl=120, alpha=0.995, extra=0.0, seeds=(0,), orders=("lb_desc",)):
    with open(path) as f:
        txt = f.read()
    inst = json.loads(txt)
    plan = build_and_solve(inst, alpha_free=alpha, alpha_tot=alpha, extra=extra, time_limit=tl)
    print(f"  MIP obj {plan['obj']:.0f} full {len(plan['full'])} half {len(plan['half'])} "
          f"src {len(plan['source'])}", flush=True)
    best = (-1, None)
    for order in orders:
        for seed in seeds:
            dm, failed = assign(inst, plan, seed=seed, order=order)
            dm, mm, full, half, us = finalize(inst, plan, dm, failed)
            pre = sum(inst["models"][i]["value"] for i in full) + sum(inst["models"][i]["value"] // 2 for i in half)
            added = greedy_add(inst, dm, mm, full, half, us)
            sol = {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": mm}
            sc, ok, msg = score_of(txt, sol)
            print(f"  order {order} seed {seed}: failed {len(failed)} pre {pre} added {added} "
                  f"score {sc} {'OK' if ok else msg}", flush=True)
            if ok and sc > best[0]:
                best = (sc, sol)
    return best, plan


def save_if_better(path, sc, sol):
    os.makedirs("solutions", exist_ok=True)
    out = os.path.join("solutions", os.path.basename(path).replace(".json", "_submission.json"))
    old = -1
    if os.path.exists(out):
        with open(path) as f:
            txt = f.read()
        with open(out) as f:
            o, ok, _ = getSolutionScore(f.read(), txt, show_details=False)
        old = o if ok else -1
    if sc > old:
        with open(out, "w") as f:
            json.dump(sol, f)
        print(f"  saved {out}: {sc} (previous {old})")
    else:
        print(f"  kept {out}: {old} >= {sc}")


if __name__ == "__main__":
    path = sys.argv[1]
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 120
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.995
    (sc, sol), plan = run(path, tl, alpha)
    if sol:
        save_if_better(path, sc, sol)
