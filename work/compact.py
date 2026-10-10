"""Exact compaction steps for one type: free as much volume as possible into the pool
while every bin stays inside its window.
 A) single-item-capable bins (lb <= 10000) + pool: min-cost matching (1 item per bin)
 B) each bin: re-pack with its own items + pool items, minimal fill >= lb (subset-sum DP)
 C) pairs of bins + pool via CP-SAT (optional)
"""
import json, sys, random, collections
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, '.')
from sol6 import State


def type_bins(st, t):
    M = st.M
    full = {m for m in st.used if m not in st.src and st.is_full(m)}
    cov = st.covered()
    bins = {}
    for m in st.used:
        r = M[m]["dataRequirements"].get(t)
        if not r:
            continue
        if m not in st.src and (m, t) in cov:
            continue
        bins[(m, t)] = (r["lowerBound"], r["upperBound"], m in full)
    return bins


def pool_items(st, t):
    return [d["id"] for d in st.D if d["dataType"] == t and d["id"] not in st.item_bin]


def move(st, i, b):
    old = st.item_bin.get(i)
    if old is not None:
        st.bin_items[old].discard(i)
        del st.item_bin[i]
    if b is not None:
        st.item_bin[i] = b
        st.bin_items[b].add(i)


def step_singles(st, t, bins, freeonly):
    """re-match bins with lb<=10000 of the given pool kind (freeonly True/False)."""
    D = st.D
    sel = [k for k, (lb, ub, fo) in bins.items() if fo == freeonly and lb <= 10000]
    items = set(pool_items(st, t))
    for k in sel:
        items |= st.bin_items[k]
    items = [i for i in items if (not freeonly) or not D[i]["isCopyrighted"]]
    if not sel or not items:
        return 0
    sizes = np.array([D[i]["size"] for i in items])
    BIG = 10 ** 9
    C = np.full((len(sel), len(items)), BIG, dtype=np.int64)
    for r, k in enumerate(sel):
        lb, ub, fo = bins[k]
        ok = (sizes >= lb) & (sizes <= ub)
        C[r, ok] = sizes[ok] - lb
    rows, cols = linear_sum_assignment(C)
    before = sum(D[i]["size"] for k in sel for i in st.bin_items[k])
    newc = {}
    for r, c in zip(rows, cols):
        if C[r, c] < BIG:
            newc[sel[r]] = [items[c]]
    left = set(items) - set(x[0] for x in newc.values())
    for k in sel:
        if k in newc:
            continue
        lb, ub, fo = bins[k]
        cands = sorted(((D[i]["size"], i) for i in left if D[i]["size"] <= ub), reverse=True)[:80]
        res = min_subset(cands, lb, ub)
        if res is None:
            return 0
        newc[k] = res
        left -= set(res)
    newvol = sum(D[i]["size"] for k in sel for i in newc[k])
    if newvol >= before:
        return 0
    for k in sel:
        for i in list(st.bin_items[k]):
            move(st, i, None)
    for k, its in newc.items():
        for i in its:
            move(st, i, k)
    after = sum(D[i]["size"] for k in sel for i in st.bin_items[k])
    return before - after


def min_subset(cands, lb, ub):
    """cands: list of (size, id). minimal sum >= lb and <= ub. returns list of ids or None."""
    # bitset DP with parent tracking via list of reachable sets per item
    reach = 1
    hist = []
    mask = (1 << (ub + 1)) - 1
    for s, i in cands:
        hist.append(reach)
        reach = (reach | (reach << s)) & mask
    r = reach >> lb
    if r == 0:
        return None
    v = lb + ((r & -r).bit_length() - 1)
    # backtrack
    out = []
    for idx in range(len(cands) - 1, -1, -1):
        s, i = cands[idx]
        prev = hist[idx]
        if (prev >> v) & 1:
            continue
        out.append(i); v -= s
    assert v == 0
    return out


def step_bins(st, t, bins, rng, max_pool=60):
    D = st.D
    gained = 0
    keys = list(bins)
    rng.shuffle(keys)
    for k in keys:
        lb, ub, fo = bins[k]
        own = list(st.bin_items[k])
        cur = sum(D[i]["size"] for i in own)
        if cur == lb:
            continue
        pool = [i for i in pool_items(st, t) if (not fo) or not D[i]["isCopyrighted"]]
        # candidate pool items: smaller than ub, sample
        pool = [i for i in pool if D[i]["size"] <= ub]
        if len(pool) > max_pool:
            pool = rng.sample(pool, max_pool)
        cands = [(D[i]["size"], i) for i in own + pool]
        res = min_subset(cands, lb, min(ub, cur - 1))
        if res is None:
            continue
        new = sum(D[i]["size"] for i in res)
        for i in own:
            move(st, i, None)
        for i in res:
            move(st, i, k)
        gained += cur - new
    return gained


def step_pairs(st, t, bins, rng, npairs=200, max_pool=40, tlim=0.5, group=3):
    """re-pack small random groups of bins (+ pool sample) exactly, minimizing used volume."""
    from ortools.sat.python import cp_model
    D = st.D
    keys = list(bins)
    gained = 0
    waste = {k: sum(D[i]["size"] for i in st.bin_items[k]) - bins[k][0] for k in keys}
    hi = [k for k in keys if waste[k] > 0]
    for _ in range(npairs):
        if not hi:
            break
        g = [rng.choice(hi)] + rng.sample(keys, group - 1)
        g = list(dict.fromkeys(g))
        own = set()
        for k in g:
            own |= st.bin_items[k]
        pool = pool_items(st, t)
        if len(pool) > max_pool:
            pool = rng.sample(pool, max_pool)
        items = sorted(own | set(pool))
        mdl = cp_model.CpModel()
        x = {}
        for bi, k in enumerate(g):
            lb, ub, fo = bins[k]
            for i in items:
                if D[i]["size"] <= ub and (not fo or not D[i]["isCopyrighted"]):
                    x[i, bi] = mdl.NewBoolVar("")
        byi = collections.defaultdict(list)
        for (i, bi), v in x.items():
            byi[i].append(v)
        for i, vs in byi.items():
            mdl.AddAtMostOne(vs)
        for bi, k in enumerate(g):
            lb, ub, fo = bins[k]
            e = sum(D[i]["size"] * v for (i, b2), v in x.items() if b2 == bi)
            mdl.Add(e >= lb); mdl.Add(e <= ub)
        used = sum(D[i]["size"] * v for (i, bi), v in x.items())
        mdl.Minimize(used)
        for (i, bi), v in x.items():
            mdl.AddHint(v, 1 if st.item_bin.get(i) == g[bi] else 0)
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = tlim
        sv.parameters.num_workers = 1
        r = sv.Solve(mdl)
        if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        before = sum(D[i]["size"] for i in own)
        if sv.Value(used) < before:
            for k in g:
                for i in list(st.bin_items[k]):
                    move(st, i, None)
            for (i, bi), v in x.items():
                if sv.Value(v):
                    move(st, i, g[bi])
            gained += before - sv.Value(used)
            for k in g:
                waste[k] = sum(D[i]["size"] for i in st.bin_items[k]) - bins[k][0]
            hi = [k for k in keys if waste[k] > 0]
    return gained


def compact(st, t, rounds=3, seed=0, log=True, pairs=0):
    rng = random.Random(seed)
    bins = type_bins(st, t)
    D = st.D
    for r in range(rounds):
        g1 = step_singles(st, t, bins, True)
        g2 = step_singles(st, t, bins, False)
        g3 = step_bins(st, t, bins, rng)
        g4 = step_pairs(st, t, bins, rng, npairs=pairs) if pairs else 0
        pv = sum(D[i]["size"] for i in pool_items(st, t))
        if log:
            print(f"  {t} round {r}: singlesF {g1} singlesA {g2} bins {g3} groups {g4} pool {pv}", flush=True)
    return st


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    st = State(inst, sol)
    for t in sys.argv[3]:
        compact(st, t, rounds=int(sys.argv[4]), pairs=int(sys.argv[5]), seed=int(sys.argv[6]))
    print(st.score())
    json.dump(st.to_sol(), open(sys.argv[7], "w"))
