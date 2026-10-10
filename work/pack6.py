"""Low-waste packer for one data type.

bins: list of (key, lb, ub, freeonly)
items: list of (size, id, isfree)
Returns (assign: key -> [item ids], failed: [keys], left: [(size,id,isfree)])

Phase 1: single-item bins (lb <= T1) are matched to items by min-cost assignment
          (cost = size - lb), so they take the items closest above their lower bound.
Phase 2: multi-item bins, decreasing lb: largest items as base, then an exact closer
          (1, 2 or 3 items) with minimal excess.
"""
import bisect
import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 10 ** 9


def match_singles(bins, items, maxw=None):
    """bins: list of (key, lb, ub); items: list of (size, id, isfree).
    returns {key: item}, unmatched keys"""
    if not bins or not items:
        return {}, [b[0] for b in bins]
    sizes = np.array([it[0] for it in items], dtype=np.int64)
    nb, ni = len(bins), len(items)
    C = np.full((nb, ni), BIG, dtype=np.int64)
    for r, (k, lb, ub) in enumerate(bins):
        ok = (sizes >= lb) & (sizes <= ub)
        C[r, ok] = sizes[ok] - lb
    rows, cols = linear_sum_assignment(C)
    res, un = {}, set(b[0] for b in bins)
    for r, c in zip(rows, cols):
        if C[r, c] < BIG:
            res[bins[r][0]] = items[c]
            un.discard(bins[r][0])
    return res, list(un)


class SortedPool:
    def __init__(self, items):
        self.items = sorted(items)

    def remove(self, it):
        j = bisect.bisect_left(self.items, it)
        assert self.items[j] == it
        del self.items[j]

    def add(self, it):
        bisect.insort(self.items, it)

    def __len__(self):
        return len(self.items)

    def total(self):
        return sum(it[0] for it in self.items)


def best_closer(pool, lo, hi, max_k=3, pen3=40):
    """smallest-excess subset of 1..max_k items with sum in [lo, hi]. returns list or None
    (a 3-item closer must beat the best 1/2-item one by pen3)"""
    its = pool.items
    s = [it[0] for it in its]
    n = len(s)
    best = None
    # single
    j = bisect.bisect_left(s, lo)
    if j < n and s[j] <= hi:
        best = (s[j] - lo, [its[j]])
        if best[0] == 0:
            return best[1]
    if max_k >= 2 and n >= 2:
        # two pointers: a + b in [lo, lo + bestexcess)
        i, k = 0, n - 1
        while i < k:
            v = s[i] + s[k]
            if v < lo:
                i += 1
            else:
                if v <= hi and (best is None or v - lo < best[0]):
                    best = (v - lo, [its[i], its[k]])
                    if best[0] == 0:
                        return best[1]
                k -= 1
    if best is not None:
        best = (best[0] + pen3, best[1])
    if max_k >= 3 and n >= 3 and (best is None or best[0] > pen3):
        # a fixed, then two pointers; limit a to a sample to bound cost
        cand = range(n) if n <= 400 else np.linspace(0, n - 1, 400).astype(int)
        for a in cand:
            lo2, hi2 = lo - s[a], hi - s[a]
            i, k = 0, n - 1
            while i < k:
                if i == a:
                    i += 1; continue
                if k == a:
                    k -= 1; continue
                v = s[i] + s[k]
                if v < lo2:
                    i += 1
                else:
                    if v <= hi2 and (best is None or v - lo2 < best[0]):
                        best = (v - lo2, [its[a], its[i], its[k]])
                        if best[0] == 0:
                            return best[1]
                    k -= 1
    return None if best is None else best[1]


def fill_multi(bins, pool, close_target=9000, max_excess=None, rng=None):
    """bins: (key, lb, ub). Greedy base with largest items, exact closer."""
    res, failed = {}, []
    for k, lb, ub in sorted(bins, key=lambda b: -b[1]):
        taken = []
        deficit = lb
        # base: largest items while deficit stays above close_target
        while len(pool) and deficit - pool.items[-1][0] >= close_target:
            it = pool.items[-1]
            pool.remove(it); taken.append(it); deficit -= it[0]
        if deficit <= 0:
            res[k] = taken
            continue
        hi = ub - (lb - deficit)
        cl = best_closer(pool, deficit, hi)
        if cl is None:
            # take more base then retry
            while len(pool) and deficit > 0 and cl is None:
                it = pool.items[-1]
                if it[0] >= deficit:
                    break
                pool.remove(it); taken.append(it); deficit -= it[0]; hi = ub - (lb - deficit)
                cl = best_closer(pool, deficit, hi)
        if cl is None:
            for it in taken:
                pool.add(it)
            failed.append(k)
            continue
        for it in cl:
            pool.remove(it)
        res[k] = taken + cl
    return res, failed


def pack_pool(bins, items, T1=9000, close_target=9000):
    """bins: (key, lb, ub). All items usable. returns res, failed, leftover pool"""
    singles = [b for b in bins if b[1] <= T1]
    multis = [b for b in bins if b[1] > T1]
    sres, sun = match_singles(singles, items)
    used = set(it[1] for it in sres.values())
    pool = SortedPool([it for it in items if it[1] not in used])
    res = {k: [it] for k, it in sres.items()}
    # unmatched singles become multi bins (2 small items)
    mb = multis + [b for b in singles if b[0] in set(sun)]
    mres, mfail = fill_multi(mb, pool, close_target=close_target)
    res.update(mres)
    return res, mfail, pool


def pack_type(bins, items, T1=9000, close_target=9000):
    """bins: (key, lb, ub, freeonly); items: (size, id, isfree)."""
    bf = [(k, lb, ub) for k, lb, ub, fo in bins if fo]
    ba = [(k, lb, ub) for k, lb, ub, fo in bins if not fo]
    free = [it for it in items if it[2]]
    copy = [it for it in items if not it[2]]
    r1, f1, left_free = pack_pool(bf, free, T1, close_target)
    r2, f2, left = pack_pool(ba, copy + left_free.items, T1, close_target)
    r1.update(r2)
    return r1, f1 + f2, left
