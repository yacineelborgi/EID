"""Plan -> exact packing of free bins -> half/source bins -> finalize -> greedy -> save.
usage: python med_solve.py datasets/X.json plans/P.json [orders...]"""
import json
import os
import sys

from assign import TYPES, Pool, fill_bin
from exact_pack import dp_min_subset, pack_type
from solve import finalize, greedy_add, score_of, strict_ok


def build(inst, plan, order="lb_asc", seed=0, verbose=True):
    M, D = inst["models"], inst["datasets"]
    full = set(plan["full"]); half = set(plan["half"]); source = set(plan["source"])
    srcreq = {(i, t) for i, t in plan["srcreq"]}
    data_map = {}
    failed = set()
    free_left = {}
    fail_bins = []
    done = set()
    for t in TYPES:
        bins = [(i, M[i]["dataRequirements"][t]["lowerBound"], M[i]["dataRequirements"][t]["upperBound"])
                for i in sorted(full) if t in M[i]["dataRequirements"]]
        items = [(d["size"], d["id"]) for d in D if d["dataType"] == t and not d["isCopyrighted"]]
        fill, still, pool = pack_type(bins, items, seed=seed, order=order)
        # sacrifice: a failed bin takes the free items of a cheap full model (demoted later)
        still = list(still)
        for i in list(still):
            r_i = M[i]["dataRequirements"][t]
            if r_i["upperBound"] < r_i["lowerBound"]:
                continue
            cands = sorted((j for j in fill if j != i),
                           key=lambda j: (0 if j in failed else M[j]["value"] - M[j]["value"] // 2,
                                          -sum(x[0] for x in fill[j][2])))
            for j in cands[:300]:
                avail = set(pool) | set(fill[j][2])
                r = dp_min_subset(list(avail), r_i["lowerBound"], r_i["upperBound"])
                if r is not None:
                    fill[i] = (r_i["lowerBound"], r_i["upperBound"], r)
                    pool = avail - set(r)
                    del fill[j]
                    still.remove(i)
                    still.append(j)
                    if verbose:
                        print(f"   {t}: bin of {i} (v {M[i]['value']}) saved by sacrificing {j} "
                              f"(v {M[j]['value']})", flush=True)
                    break
        for i, (lb, ub, its) in fill.items():
            done.add((i, t))
            for it in its:
                data_map[it[1]] = i
        failed |= set(still)
        fail_bins += [(i, t) for i in still]
        free_left[t] = pool
        if verbose:
            waste = sum(sum(x[0] for x in v[2]) - v[0] for v in fill.values())
            print(f"   {t}: full bins {len(bins)} failed {len(still)} waste {waste} "
                  f"left {sum(x[0] for x in pool)}", flush=True)
    copy = {t: Pool([(d["size"], d["id"]) for d in D if d["dataType"] == t and d["isCopyrighted"]])
            for t in TYPES}
    free = {t: Pool(list(free_left[t])) for t in TYPES}
    # demote failed full models to half: fill the failed bins with copyrighted data
    demoted = set()
    for i in sorted(failed, key=lambda i: -M[i]["value"]):
        bins = [(t, M[i]["dataRequirements"][t]) for ii, t in fail_bins if ii == i]
        taken = []
        ok = True
        for t, r in bins:
            res = fill_bin(r["lowerBound"], r["upperBound"], [copy[t], free[t]])
            if res is None:
                ok = False
                break
            taken += res
        if ok:
            for _, it in taken:
                data_map[it[1]] = i
            demoted.add(i)
            done.update((i, t) for t, _ in bins)
        else:
            for p, it in taken:
                p.put(it)
    failed -= demoted
    if demoted:
        plan = dict(plan)
        plan["full"] = [i for i in plan["full"] if i not in demoted]
        plan["half"] = list(plan["half"]) + sorted(demoted)
        half = set(plan["half"])
        if verbose:
            print("   demoted to half:", sorted(demoted), flush=True)
    for t in TYPES:
        bins = []
        for i in half | source:
            r = M[i]["dataRequirements"].get(t)
            if r and (i, t) not in srcreq and (i, t) not in done:
                bins.append((i, r["lowerBound"], r["upperBound"]))
        bins.sort(key=lambda b: -b[1])
        for i, lb, ub in bins:
            if i in failed or ub < lb:
                failed.add(i)
                continue
            res = fill_bin(lb, ub, [copy[t], free[t]])
            if res is None:
                failed.add(i)
            else:
                for _, it in res:
                    data_map[it[1]] = i
    dm, mm, f2, h2, us = finalize(inst, plan, data_map, failed)
    pre = sum(M[i]["value"] for i in f2) + sum(M[i]["value"] // 2 for i in h2)
    added = greedy_add(inst, dm, mm, f2, h2, us)
    sol = {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": mm}
    return sol, failed, pre, added


if __name__ == "__main__":
    path, pf = sys.argv[1], sys.argv[2]
    orders = sys.argv[3:] or ["lb_asc"]
    txt = open(path).read()
    inst = json.loads(txt)
    plan = json.load(open(pf))
    print("plan obj", plan["obj"], "bound", plan.get("bound"))
    best = (-1, None)
    for o in orders:
        sol, failed, pre, added = build(inst, plan, order=o)
        sc, ok, msg = score_of(txt, sol)
        ok2 = ok and strict_ok(inst, sol)
        print(f" order {o}: failed {sorted(failed)} pre {pre} added {added} score {sc} "
              f"{'OK' if ok2 else msg}", flush=True)
        if ok2 and sc > best[0]:
            best = (sc, sol)
    if best[1]:
        os.makedirs("cands", exist_ok=True)
        out = os.path.join("cands", os.path.basename(pf).replace(".json", f"_{best[0]}.json"))
        json.dump(best[1], open(out, "w"))
        print("BEST", best[0], out)
