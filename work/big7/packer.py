"""Exact packer for 7_big plans.
Phase A (per type): full bins, free datasets only, lb ascending: smallest-dataset for tiny bins,
exact single / pair / DP closing, greedy big items for large bins, end-game DP.
Phase B: half + source bins with copyrighted datasets (+ free leftovers).
"""
import sys, random
import numpy as np
from sortedcontainers import SortedList
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
sys.path.insert(0, "/home/user/EID")
from exact_assign import dp_close
BIG = 1 << 60

class Pool:
    def __init__(self, items):
        self.sl = SortedList(items)  # (size, id)
    def total(self):
        return sum(s for s, _ in self.sl)
    def remove(self, it):
        self.sl.remove(it)
    def add(self, it):
        self.sl.add(it)
    def exact(self, g, rng=None):
        i = self.sl.bisect_left((g, -1))
        if i < len(self.sl) and self.sl[i][0] == g:
            return self.sl[i]
        return None
    def ge(self, g):
        i = self.sl.bisect_left((g, -1))
        return self.sl[i] if i < len(self.sl) else None
    def le(self, g):
        i = self.sl.bisect_right((g, BIG))
        return self.sl[i - 1] if i > 0 else None
    def pair(self, g, rng, tries=400):
        """two distinct items summing exactly to g."""
        sl = self.sl
        n = sl.bisect_right((g, BIG))
        if n == 0:
            return None
        lo = sl.bisect_left(((g + 1) // 2, -1))  # a >= g/2 -> b <= a
        cand = range(lo, n)
        if len(cand) > tries:
            cand = rng.sample(cand, tries)
        for k in cand:
            a = sl[k]
            b = g - a[0]
            if b <= 0:
                continue
            j = sl.bisect_left((b, -1))
            while j < len(sl) and sl[j][0] == b:
                if sl[j][1] != a[1]:
                    return [a, sl[j]]
                j += 1
        return None
    def sample(self, hi, K, rng):
        sl = self.sl
        n = sl.bisect_right((hi, BIG))
        if n <= K:
            return [sl[j] for j in range(n)]
        idx = sorted(rng.sample(range(n), K))
        return [sl[j] for j in idx]

def fill_bin(pool, lb, ub, rng, max_waste, G0=12000, K=400):
    """returns list of items (removed from pool) or None."""
    if ub < lb:
        return None
    sl = pool.sl
    if not sl:
        return None
    # tiny: smallest item already >= lb
    if sl[0][0] >= lb:
        it = pool.ge(lb)
        if it[0] - lb <= max_waste and it[0] <= ub:
            pool.remove(it); return [it]
        return None
    it = pool.exact(lb)
    if it is not None:
        pool.remove(it); return [it]
    taken = []
    g = lb
    while g > G0:
        it = pool.le(g - G0 // 2)
        if it is None:
            break
        pool.remove(it); taken.append(it); g -= it[0]
    res = None
    if g <= 0:
        res = []
    else:
        it = pool.exact(g)
        if it is not None:
            res = [it]
        else:
            pr = pool.pair(g, rng)
            if pr is not None:
                res = pr
            else:
                cands = sorted(set(pool.sample(g, K, rng)), key=lambda c: -c[0])
                res = dp_close(cands, g, g)
                if res is None:
                    hi = min(ub - (lb - g), g + max_waste)
                    cands = sorted(set(pool.sample(hi, 3 * K, rng)), key=lambda c: -c[0])
                    res = dp_close(cands, g, hi)
    if res is None:
        for it in taken:
            pool.add(it)
        return None
    for it in res:
        pool.remove(it)
    return taken + res

def pack_type(bins, items, rng, max_waste_total=10**9, order="lb_asc", G0=12000):
    """bins: list (model, lb, ub). returns dict model->items, failed list, waste."""
    pool = Pool(items)
    if order == "lb_asc":
        bins = sorted(bins, key=lambda b: (b[1], b[2]))
    elif order == "random":
        bins = bins[:]; rng.shuffle(bins)
    out = {}; failed = []; waste = 0
    for m, lb, ub in bins:
        res = fill_bin(pool, lb, ub, rng, max(0, max_waste_total - waste), G0=G0)
        if res is None:
            failed.append((m, lb, ub)); continue
        out[m] = res
        waste += sum(s for s, _ in res) - lb
    return out, failed, waste, pool
