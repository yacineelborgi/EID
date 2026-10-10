"""Near-zero-waste packing of free datasets into the bins of `full` models.

Bins are processed one at a time; each one is closed with a subset of the
remaining items whose sum is the smallest reachable value >= lb (bitset DP over
the whole remaining pool, preferring big items so small ones stay available for
fine tuning). Failed bins are repaired by re-packing them together with a few
already-filled bins (pooled items + DP).
"""
import random


def dp_min_subset(items, lo, hi, prefer_big=True):
    """items: list of (size, id). Subset with minimal sum in [lo, hi]; None if none."""
    if hi < 0:
        return None
    if lo <= 0:
        return []
    its = sorted(items, key=lambda x: -x[0]) if prefer_big else list(items)
    its = [x for x in its if x[0] <= hi]
    mask = (1 << (hi + 1)) - 1
    reach = 1
    snaps = []
    for it in its:
        snaps.append(reach)
        reach = (reach | (reach << it[0])) & mask
    top = reach >> lo
    if not top:
        return None
    S = lo + ((top & -top).bit_length() - 1)
    out = []
    for k in range(len(its) - 1, -1, -1):
        if S == 0:
            break
        if not (snaps[k] >> S) & 1:
            out.append(its[k])
            S -= its[k][0]
    assert S == 0
    return out


def pack_type(bins, items, seed=0, order="lb_desc", repair_rounds=200, K_rep=3):
    """bins: list of (key, lb, ub); items: list of (size, id).
    Returns (fill: key -> list of items, failed: list of keys, leftover items)."""
    rnd = random.Random(seed)
    pool = set(items)
    fill = {}
    failed = []
    bl = list(bins)
    if order == "lb_desc":
        bl.sort(key=lambda b: -b[1])
    elif order == "lb_asc":
        bl.sort(key=lambda b: (b[1], b[2]))
    elif order == "ub_asc":
        bl.sort(key=lambda b: (b[2], b[1]))
    elif order == "random":
        rnd.shuffle(bl)
    elif order == "width":
        bl.sort(key=lambda b: (b[2] - b[1], -b[1]))
    for key, lb, ub in bl:
        if ub < lb:
            failed.append((key, lb, ub))
            continue
        res = dp_min_subset(list(pool), lb, ub)
        if res is None:
            failed.append((key, lb, ub))
            continue
        fill[key] = (lb, ub, res)
        pool.difference_update(res)
    # repair: for a failed bin, pool its needs with some filled bins and re-pack jointly
    binfo = {k: (lb, ub) for k, lb, ub in bl}
    still = []
    for key, lb, ub in failed:
        if ub < lb:
            still.append(key)
            continue
        ok = False
        keys = list(fill.keys())
        for _ in range(repair_rounds):
            others = rnd.sample(keys, min(K_rep, len(keys)))
            avail = set(pool)
            for o in others:
                avail.update(fill[o][2])
            # fill the failed bin first (prefer big items), then others exactly
            tries = [(key, lb, ub)] + [(o, fill[o][0], fill[o][1]) for o in others]
            newf = {}
            av = set(avail)
            good = True
            for kk, l, u in tries:
                r = dp_min_subset(list(av), l, u)
                if r is None:
                    good = False
                    break
                newf[kk] = (l, u, r)
                av.difference_update(r)
            if good:
                for kk, v in newf.items():
                    fill[kk] = v
                pool = av
                ok = True
                break
        if not ok:
            still.append(key)
    return fill, still, pool
