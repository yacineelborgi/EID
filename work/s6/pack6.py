"""Patched assignment: bins sorted by (lb with noise, ub) so narrow tiny bins go first."""
import sys
sys.path.insert(0, '/home/user/EID')
import solve2
from assign import TYPES, Pool, fill_bin
TINY = 3000
G0 = 22000


def assign6(inst, plan, rnd, noise=0.0, exclude=(), order="lb_asc"):
    M, D = inst["models"], inst["datasets"]
    full = set(plan["full"]); half = set(plan["half"]); source = set(plan["source"])
    srcreq = {(i, t) for i, t in plan["srcreq"]}
    exclude = set(exclude)
    free = {t: Pool([]) for t in TYPES}; copy = {t: Pool([]) for t in TYPES}
    for d in D:
        if d["id"] in exclude: continue
        (copy if d["isCopyrighted"] else free)[d["dataType"]].put((d["size"], d["id"]))
    data_map = {}; failed = set()
    for t in TYPES:
        bins_free, bins_any = [], []
        for i in full:
            r = M[i]["dataRequirements"].get(t)
            if r: bins_free.append((i, r["lowerBound"], r["upperBound"]))
        for i in half | source:
            r = M[i]["dataRequirements"].get(t)
            if r and (i, t) not in srcreq: bins_any.append((i, r["lowerBound"], r["upperBound"]))
        if order == "mix":
            key = lambda b: ((1, -b[1] * (1 + noise * rnd.random()), 0) if b[1] > TINY else (0, b[1], b[2]))
        else:
            key = lambda b: ((b[1] * (1 + noise * rnd.random())) if b[1] > TINY else b[1], b[2])
        bins_free.sort(key=key); bins_any.sort(key=key)
        for bins, pools in ((bins_free, [free[t]]), (bins_any, [copy[t], free[t]])):
            for i, lb, ub in bins:
                if i in failed or ub < lb:
                    failed.add(i); continue
                res = fill_bin(lb, ub, pools, G0=G0)
                if res is None: failed.add(i)
                else:
                    for _, it in res: data_map[it[1]] = i
    return data_map, failed, free, copy


solve2.assign2 = assign6
