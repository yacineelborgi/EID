"""Phase 2: turn a plan (models + modes) into concrete dataset / model mappings.

Each data type is independent. A "bin" is one requirement [lb, ub] of one model
that must be filled with datasets of that type. Bins of `full` models may only
use free datasets; all other bins prefer copyrighted datasets.
"""
import random

from sortedcontainers import SortedList

TYPES = "ntic"
TOL_DIV = 100  # accepted waste (window/TOL_DIV) before trying the subset-sum DP


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


def _dp_close(cands, g, hi):
    """Subset of cands (list of (pool,item)) with minimal sum in [g, hi]."""
    if hi < 0 or not cands:
        return None
    mask = (1 << (hi + 1)) - 1
    reach = 1
    snaps = []
    for _, it in cands:
        snaps.append(reach)
        reach = (reach | (reach << it[0])) & mask
    top = reach >> g
    if not top:
        return None
    S = g + ((top & -top).bit_length() - 1)
    out = []
    for k in range(len(cands) - 1, -1, -1):
        if S == 0:
            break
        if not (snaps[k] >> S) & 1:
            out.append(cands[k])
            S -= cands[k][1][0]
    return out


def _sample(p, hi, K):
    n = p.s.bisect_right((hi, 1 << 60))
    if n <= K:
        return [(p, p.s[j]) for j in range(n)]
    step = n / K
    return [(p, p.s[int(j * step)]) for j in range(K)]


def fill_bin(lb, ub, pools, K=300, G0=22000):
    """Fill [lb, ub] with minimal waste. Returns list of (pool, item) or None."""
    taken = []
    cur = 0
    # phase 1: big items while far from the target
    while lb - cur > G0:
        pick = None
        for p in pools:
            it = p.largest_le(min(lb - cur - G0 + 10000, ub - cur))
            if it is not None:
                pick = (p, it)
                break
        if pick is None:
            break
        pick[0].take(pick[1])
        taken.append(pick)
        cur += pick[1][0]
    g, hi = max(lb - cur, 0), ub - cur
    res = None
    if g == 0:
        res = []
    else:
        # quick single-item close
        for p in pools:
            it = p.smallest_ge(g)
            if it is not None and it[0] - g <= max(2, (hi - g) // TOL_DIV) and it[0] <= hi:
                res = [(p, it)]
                break
        if res is None:
            cands = []
            for p in pools:
                cands += _sample(p, hi, K)
                # big items first: backtracking then only uses small items when needed
                res = _dp_close(sorted(cands, key=lambda c: -c[1][0]), g, hi)
                if res is not None and sum(it[0] for _, it in res) - g <= max(2, (hi - g) // TOL_DIV):
                    break
    if res is None:
        for p, it in taken:
            p.put(it)
        return None
    for p, it in res:
        p.take(it)
    return taken + res


def assign(inst, plan, seed=0, order="lb_desc", exclude=()):
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
    exclude = set(exclude)
    for d in D:
        if d["id"] in exclude:
            continue
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
        key = {"width": lambda b: (b[2] - b[1], -b[1]), "lb_asc": lambda b: b[1]}.get(order, lambda b: -b[1])
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
