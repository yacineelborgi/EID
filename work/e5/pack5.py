"""Packer for 5_energy plans.

Per type: phase A (full bins, free items), phase B (half + source bins, copyrighted + free leftovers).
In each phase:
  1. tiny bins (lb <= theta) get ONE item each through a matroid greedy (items ascending, the pending
     bin with the smallest ub first) -> minimal total of used items, respects ub;
  2. other bins in lb ascending order: big items first, exact subset-sum closing (exact_assign.Packer).
"""
import heapq
import json
import random
import sys
import time

sys.path.insert(0, "/home/user/EID")
from sortedcontainers import SortedList

from exact_assign import Packer
from solve import finalize, greedy_add, score_of, strict_ok

TYPES = "ntic"


def tiny_match(bins, pool):
    """bins: list (i, lb, ub). pool: SortedList of (size, id). Returns (assign dict i->item, failed list)."""
    bins = sorted(bins, key=lambda b: b[1])
    out = {}
    failed = []
    heap = []
    k = 0
    pos = 0
    while (k < len(bins) or heap) and pos < len(pool):
        s = pool[pos]
        while k < len(bins) and bins[k][1] <= s[0]:
            heapq.heappush(heap, (bins[k][2], bins[k][0], bins[k][1]))
            k += 1
        while heap and heap[0][0] < s[0]:
            ub, i, lb = heapq.heappop(heap)
            failed.append((i, lb, ub))
        if heap:
            ub, i, lb = heapq.heappop(heap)
            out[i] = s
        pos += 1
    failed += [(i, lb, ub) for ub, i, lb in heap] + [(b[0], b[1], b[2]) for b in bins[k:]]
    for s in out.values():
        pool.remove(s)
    return out, failed


def pack_phase(bins, items, theta, rng, K=300, G0=20000, order="lb_asc"):
    pool = SortedList(items)
    tiny = [b for b in bins if b[1] <= theta]
    rest = [b for b in bins if b[1] > theta]
    asg, failed = tiny_match(tiny, pool)
    res = {i: [it] for i, it in asg.items()}
    waste = sum(it[0] - lb for (i, lb, ub) in tiny if i in asg for it in [asg[i]])
    pk = Packer([], rng, K=K, G0=G0)
    pk.pool = pool
    if order == "lb_asc":
        rest.sort(key=lambda b: b[1])
    elif order == "lb_desc":
        rest.sort(key=lambda b: -b[1])
    elif order == "random":
        rng.shuffle(rest)
    for i, lb, ub in rest:
        r = pk.fill(lb, ub, ub - lb)
        if r is None:
            failed.append((i, lb, ub))
            continue
        res[i] = r
        waste += sum(s for s, _ in r) - lb
    return res, failed, waste, list(pk.pool)


def pack(inst, plan, theta=300, seed=0, order="lb_asc", verbose=False):
    M, D = inst["models"], inst["datasets"]
    rng = random.Random(seed)
    full, half, source = set(plan["full"]), set(plan["half"]), set(plan["source"])
    srcreq = {(i, t) for i, t in plan["srcreq"]}
    data_map = {}
    failed = set()
    stats = {}
    for t in TYPES:
        fr = [(d["size"], d["id"]) for d in D if d["dataType"] == t and not d["isCopyrighted"]]
        cp = [(d["size"], d["id"]) for d in D if d["dataType"] == t and d["isCopyrighted"]]
        bA = [(i, M[i]["dataRequirements"][t]["lowerBound"], M[i]["dataRequirements"][t]["upperBound"])
              for i in full if t in M[i]["dataRequirements"]]
        bB = [(i, M[i]["dataRequirements"][t]["lowerBound"], M[i]["dataRequirements"][t]["upperBound"])
              for i in half | source if t in M[i]["dataRequirements"] and (i, t) not in srcreq]
        resA, fA, wA, left = pack_phase(bA, fr, theta, rng, order=order)
        resB, fB, wB, left2 = pack_phase(bB, cp + left, theta, rng, order=order)
        for res in (resA, resB):
            for i, its in res.items():
                for _, d in its:
                    data_map[d] = i
        failed |= {b[0] for b in fA + fB}
        stats[t] = dict(nA=len(bA), nB=len(bB), wA=wA, wB=wB, fA=fA, fB=fB, leftA=sum(s for s, _ in left),
                        left=sum(s for s, _ in left2))
        if verbose:
            print("  ", t, stats[t], flush=True)
    data_map = {d: m for d, m in data_map.items() if m not in failed}
    return data_map, failed, stats


def evaluate(inst, txt, plan, theta=300, seeds=(0,), order="lb_asc", verbose=True):
    M = inst["models"]
    best = (-1, None)
    for seed in seeds:
        t0 = time.time()
        dm, failed, st = pack(inst, plan, theta, seed, order, verbose=verbose)
        dm2, mm, full, half, us = finalize(inst, plan, dm, failed)
        pre = sum(M[i]["value"] for i in full) + sum(M[i]["value"] // 2 for i in half)
        added = greedy_add(inst, dm2, mm, full, half, us)
        sol = {"dataMappings": [[d, m] for d, m in dm2.items()], "modelMappings": mm}
        sc, ok, msg = score_of(txt, sol)
        ok = ok and strict_ok(inst, sol)
        print(f"  theta {theta} seed {seed}: failed {len(failed)} pre {pre} added {added} score {sc} {ok} {msg} "
              f"{time.time() - t0:.0f}s", flush=True)
        if ok and sc > best[0]:
            best = (sc, sol)
    return best


if __name__ == "__main__":
    INST = "/home/user/EID/datasets/5_energy.json"
    txt = open(INST).read()
    inst = json.loads(txt)
    plan = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    thetas = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [300]
    import os
    old = -1
    if os.path.exists(out):
        old, ok, _ = score_of(txt, json.load(open(out)))
    print("plan obj", plan.get("obj"), "old", old)
    for th in thetas:
        sc, sol = evaluate(inst, txt, plan, th)
        if sol and sc > old:
            json.dump(sol, open(out, "w"))
            old = sc
            print("  wrote", out, sc)
