"""Phase 2: turn a plan (models + modes) into concrete dataset / model mappings.

Each data type is independent. A "bin" is one requirement [lb, ub] of one model
that must be filled with datasets of that type. Bins of `full` models may only
use free datasets; all other bins prefer copyrighted datasets.
"""
import random

from sortedcontainers import SortedList

TYPES = "ntic"


class Pool:
    def __init__(self, items):
        self.s = SortedList(items)  # (size, id)

    def __len__(self):
        return len(self.s)

    def take(self, it):
        self.s.remove(it)

    def put(self, it):
        self.s.add(it)

    def smallest_ge(self, x):
        i = self.s.bisect_left((x, -1))
        return self.s[i] if i < len(self.s) else None

    def largest_le(self, x):
        i = self.s.bisect_right((x, 1 << 60))
        return self.s[i - 1] if i > 0 else None


def try_close(pools, g, w, two_tries=60):
    """Find 1 or 2 items (pools in order of preference) with sum in [g, g+w]."""
    for p in pools:
        it = p.smallest_ge(g)
        if it is not None and it[0] <= g + w:
            return [(p, it)]
    for p in pools:
        for q in pools:
            i = p.s.bisect_right((g, 1 << 60))
            for j in range(i - 1, max(0, i - two_tries) - 1, -1):
                a = p.s[j]
                k = q.s.bisect_left((g - a[0], -1))
                while k < len(q.s) and q.s[k] == a:
                    k += 1
                if k < len(q.s) and a[0] + q.s[k][0] <= g + w:
                    return [(p, a), (q, q.s[k])]
    return None


def fill_bin(lb, ub, pools, reserve=5000):
    """Greedy fill. Returns list of (pool, item) or None (and restores pools)."""
    taken = []
    cur = 0
    w = ub - lb
    while True:
        g = lb - cur
        if g <= 0:
            return taken
        if not any(len(p) for p in pools):
            break
        res = try_close(pools, g, w)
        if res is not None:
            for p, it in res:
                p.take(it)
            taken += res
            return taken
        # take one item leaving roughly `reserve` to close
        target = g - reserve if g > reserve else g // 2
        pick = None
        for p in pools:
            it = p.largest_le(max(target, 1))
            if it is not None:
                pick = (p, it)
                break
        if pick is None:
            break
        pick[0].take(pick[1])
        taken.append(pick)
        cur += pick[1][0]
    for p, it in taken:
        p.put(it)
    return None


def assign(inst, plan, seed=0, order="lb_desc"):
    """Returns (dataMappings, modelMappings, failed_models)."""
    M = inst["models"]
    D = inst["datasets"]
    rnd = random.Random(seed)
    full = set(plan["full"])
    half = set(plan["half"])
    source = set(plan["source"])
    srcreq = {(i, t) for i, t in plan["srcreq"]}

    free = {t: Pool([]) for t in TYPES}
    copy = {t: Pool([]) for t in TYPES}
    for d in D:
        (copy if d["isCopyrighted"] else free)[d["dataType"]].put((d["size"], d["id"]))

    data_map = {}  # dataset -> model
    failed = set()
    # bins per type
    for t in TYPES:
        bins_free, bins_any = [], []
        for i in full:
            r = M[i]["dataRequirements"].get(t)
            if r:
                bins_free.append((i, r["lowerBound"], r["upperBound"]))
        for i in half | source:
            r = M[i]["dataRequirements"].get(t)
            if r and (i, t) not in srcreq:
                bins_any.append((i, r["lowerBound"], r["upperBound"]))
        key = (lambda b: (b[2] - b[1], -b[1])) if order == "width" else (lambda b: -b[1])
        if order == "random":
            rnd.shuffle(bins_free); rnd.shuffle(bins_any)
        else:
            bins_free.sort(key=key); bins_any.sort(key=key)
        for bins, pools in ((bins_free, [free[t]]), (bins_any, [copy[t], free[t]])):
            for i, lb, ub in bins:
                if i in failed or ub < lb:
                    failed.add(i)
                    continue
                res = fill_bin(lb, ub, pools)
                if res is None:
                    failed.add(i)
                else:
                    for _, it in res:
                        data_map[it[1]] = i
    return data_map, failed
