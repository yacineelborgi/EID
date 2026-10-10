"""Randomised exact filler: bins get 1..4 items with sum in [lb, lb+w], first items sampled uniformly from the
pool (keeps the pool distribution balanced), last item found by bisect."""
import random
from sortedcontainers import SortedList

BIG = 1 << 60


def find_in(pool, lo, hi, exclude=()):
    i = pool.bisect_left((lo, -1))
    while i < len(pool) and pool[i][0] <= hi:
        if pool[i] not in exclude:
            return pool[i]
        i += 1
    return None


def fill_exact(pool, L, U, w, rng, tries=60, smin=None, smax=None):
    hi = min(U, L + w)
    if not pool:
        return None
    smin = pool[0][0] if smin is None else smin
    smax = pool[-1][0] if smax is None else smax
    # single
    if L <= smax:
        it = find_in(pool, L, hi)
        if it is not None:
            return [it]
    for k in (2, 3, 4, 5):
        if L > k * smax:
            continue
        for _ in range(tries):
            chosen = []
            rem_lo, rem_hi = L, hi
            ok = True
            for r in range(k - 1):
                left = k - 1 - r  # items still to add after this one
                a_lo = max(smin, rem_lo - left * smax)
                a_hi = min(smax, rem_hi - left * smin)
                if a_lo > a_hi:
                    ok = False; break
                i0 = pool.bisect_left((a_lo, -1)); i1 = pool.bisect_right((a_hi, BIG))
                if i1 <= i0:
                    ok = False; break
                it = pool[rng.randrange(i0, i1)]
                if it in chosen:
                    ok = False; break
                chosen.append(it)
                rem_lo -= it[0]; rem_hi -= it[0]
            if not ok:
                continue
            last = find_in(pool, max(rem_lo, 0), rem_hi, chosen)
            if last is not None:
                return chosen + [last]
    return None


def seq_fill(bins, pool, rng, w=0, w2=20, order="desc"):
    """bins: list (b, L, U). pool SortedList (size, j). Returns dict b->items, failed list."""
    if order == "desc":
        bins = sorted(bins, key=lambda x: -x[1])
    elif order == "asc":
        bins = sorted(bins, key=lambda x: x[1])
    else:
        bins = bins[:]; rng.shuffle(bins)
    res, failed = {}, []
    for b, L, U in bins:
        r = fill_exact(pool, L, U, w, rng)
        if r is None and w2 > w:
            r = fill_exact(pool, L, U, w2, rng)
        if r is None:
            failed.append((b, L, U)); continue
        for it in r:
            pool.remove(it)
        res[b] = r
    return res, failed
