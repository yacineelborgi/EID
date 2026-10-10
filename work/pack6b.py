"""Packer v2: bases of large items for multi-item bins, then ONE global min-cost
matching of closers (single bins: closer = whole bin) -> minimal total waste
for the chosen bases. Several base strategies are tried, best kept."""
import bisect, random
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

BIG = 10 ** 7


def closer_match(reqs, items, win_cap=None):
    """reqs: list of (key, r, w): need one item with size in [r, r+w].
    items: list of (size,id,free). Min total (size - r); maximize number matched first.
    returns {key: item}, [unmatched keys]"""
    if not reqs:
        return {}, []
    sizes = np.array([it[0] for it in items], dtype=np.int64)
    order = np.argsort(sizes, kind="stable")
    ss = sizes[order]
    rows, cols, vals = [], [], []
    for q, (k, r, w) in enumerate(reqs):
        lo = np.searchsorted(ss, r, "left")
        hi = np.searchsorted(ss, r + w, "right")
        if win_cap is not None:
            hi = min(hi, lo + win_cap)
        for j in range(lo, hi):
            rows.append(q); cols.append(order[j]); vals.append(int(ss[j] - r) + 1)
    nq, ni = len(reqs), len(items)
    # dense LSA with big cost for infeasible: maximize matched count first
    C = np.full((nq, ni), BIG, dtype=np.int64)
    C[rows, cols] = vals
    rr, cc = linear_sum_assignment(C)
    res, un = {}, []
    for q, c in zip(rr, cc):
        if C[q, c] < BIG:
            res[reqs[q][0]] = items[c]
        else:
            un.append(reqs[q][0])
    matched_rows = set(rr)
    un += [reqs[q][0] for q in range(nq) if q not in matched_rows]
    return res, un


def make_bases(multis, items, target_lo, target_hi, rng=None, mode="largest"):
    """For each multi bin choose base items so residual in [target_lo, target_hi].
    returns {key: (base_items, residual)}, remaining items list"""
    pool = sorted(items)
    out = {}
    order = sorted(multis, key=lambda b: -b[1])
    if rng is not None and mode == "shuffle":
        rng.shuffle(order)
    for k, lb, ub in order:
        base = []
        d = lb
        while d > target_hi and pool:
            # largest item keeping residual >= target_lo
            j = bisect.bisect_right(pool, (d - target_lo, 10 ** 12, True)) - 1
            if j < 0:
                break
            if mode == "random" and rng is not None and j > 5:
                j = j - rng.randrange(min(6, j))
            it = pool.pop(j)
            base.append(it); d -= it[0]
        out[k] = (base, d)
    return out, pool


def pack_pool2(bins, items, T1=10000, tlo=300, thi=9500, rng=None, mode="largest"):
    singles = [b for b in bins if b[1] <= T1]
    multis = [b for b in bins if b[1] > T1]
    # 1. singles first, optimal min-waste matching on all items
    sres, sun = closer_match([(k, lb, ub - lb) for k, lb, ub in singles], items)
    usedS = set(it[1] for it in sres.values())
    rest0 = [it for it in items if it[1] not in usedS]
    # 2. bases for multis (and unmatched singles get no base)
    bases, rest = make_bases(multis, rest0, tlo, thi, rng, mode)
    ub_of = {k: ub for k, lb, ub in bins}
    lb_of = {k: lb for k, lb, ub in bins}
    reqs = [(k, d, ub_of[k] - lb_of[k]) for k, (base, d) in bases.items()]
    res, un = closer_match(reqs, rest)
    out = {k: [it] for k, it in sres.items()}
    used = set()
    for k, it in res.items():
        out[k] = [it] + bases[k][0]
        used.add(it[1])
    left = [it for it in rest if it[1] not in used]
    failed = list(sun)
    for k in un:
        left += bases[k][0]
        failed.append(k)
    # 3. failed bins: try 2/3-item closers from leftovers
    from pack6 import SortedPool, best_closer
    pool = SortedPool(left)
    still = []
    for k in sorted(failed, key=lambda k: -lb_of[k]):
        lb, ub = lb_of[k], ub_of[k]
        taken = []
        d = lb
        while len(pool) and d - pool.items[-1][0] >= 6000:
            it = pool.items[-1]; pool.remove(it); taken.append(it); d -= it[0]
        cl = best_closer(pool, d, ub - (lb - d)) if d > 0 else []
        if cl is None:
            for it in taken: pool.add(it)
            still.append(k); continue
        for it in cl: pool.remove(it)
        out[k] = taken + cl
    return out, still, pool.items


def pack_type2(bins, items, **kw):
    bf = [(k, lb, ub) for k, lb, ub, fo in bins if fo]
    ba = [(k, lb, ub) for k, lb, ub, fo in bins if not fo]
    free = [it for it in items if it[2]]
    copy = [it for it in items if not it[2]]
    r1, f1, lf = pack_pool2(bf, free, **kw)
    r2, f2, left = pack_pool2(ba, copy + lf, **kw)
    r1.update(r2)
    return r1, f1 + f2, left
