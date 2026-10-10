"""Exact dataset assignment: every bin [lb, ub] is filled to exactly lb whenever possible.

Per data type, two pools:
  phase A: bins of `full` models, free datasets only;
  phase B: bins of `half`/`source` models, copyrighted datasets + free leftovers.
Bins are filled one by one: big items first (leaving a gap in a comfortable range), then a
subset-sum DP (bitset) closes the gap exactly. If no exact closing exists, the smallest
waste is accepted (bounded by a budget). Failed bins are reported.
"""
import random

from sortedcontainers import SortedList

TYPES = "ntic"
BIG = 1 << 60


def dp_close(cands, g, hi):
    """cands sorted by preference (most preferred first). Minimal sum in [g, hi]."""
    if g <= 0:
        return []
    if hi < g or not cands:
        return None
    mask = (1 << (hi + 1)) - 1
    reach = 1
    snaps = []
    for it in cands:
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
            S -= cands[k][0]
    assert S == 0
    return out


class Packer:
    def __init__(self, items, rng, K=300, G0=20000, keep_small=0):
        self.pool = SortedList(items)
        self.rng = rng
        self.K = K
        self.G0 = G0
        self.keep_small = keep_small

    def total(self):
        return sum(s for s, _ in self.pool)

    def sample(self, hi, K):
        p = self.pool
        n = p.bisect_right((hi, BIG))
        if n <= K:
            return [p[j] for j in range(n)]
        step = n / K
        off = self.rng.random() * step
        return [p[min(n - 1, int(off + j * step))] for j in range(K)]

    def fill(self, lb, ub, max_waste):
        """Returns list of items or None. Items are removed from the pool."""
        p = self.pool
        taken = []
        g = lb
        while g > self.G0:
            # largest item leaving at least G0-10000 to close with the DP
            i = p.bisect_right((min(g - self.G0 + 10000, ub - (lb - g)), BIG))
            if i == 0:
                break
            it = p[i - 1]
            p.remove(it)
            taken.append(it)
            g -= it[0]
        hi = min(ub - (lb - g), g + max_waste)
        res = None
        if g <= 0:
            res = []
        else:
            # exact single item?
            i = p.bisect_left((g, -1))
            if i < len(p) and p[i][0] == g:
                res = [p[i]]
            else:
                cands = self.sample(hi, self.K)
                cands = sorted(set(cands), key=lambda c: -c[0])
                res = dp_close(cands, g, g)
                if res is None:
                    # bigger sample, all items
                    n = p.bisect_right((hi, BIG))
                    if n > len(cands):
                        cands = self.sample(hi, max(self.K * 4, 1200))
                        cands = sorted(set(cands), key=lambda c: -c[0])
                    res = dp_close(cands, g, hi)
        if res is None:
            for it in taken:
                p.add(it)
            return None
        for it in res:
            p.remove(it)
        return taken + res


def order_bins(bins, order, rng):
    if order == "random":
        bins = bins[:]
        rng.shuffle(bins)
        return bins
    key = {"width": lambda b: (b[2] - b[1], -b[1]),
           "lb_asc": lambda b: b[1],
           "lb_desc": lambda b: -b[1],
           "width_desc": lambda b: (-(b[2] - b[1]), -b[1])}[order]
    return sorted(bins, key=key)


def assign_exact(inst, plan, seed=0, order="width", K=300, G0=20000, exclude=(), budget_frac=1.0,
                 verbose=False, demote=False):
    """Returns (data_map dataset->model, failed set, stats)."""
    M = inst["models"]
    D = inst["datasets"]
    rng = random.Random(seed)
    full = set(plan["full"])
    half = set(plan["half"])
    source = set(plan["source"])
    srcreq = {(i, t) for i, t in plan["srcreq"]}
    exclude = set(exclude)
    data_map = {}
    failed_bins = []
    fset = set()
    stats = {}
    demoted = set()
    for t in TYPES:
        fr = [(d["size"], d["id"]) for d in D if d["dataType"] == t and not d["isCopyrighted"] and d["id"] not in exclude]
        cp = [(d["size"], d["id"]) for d in D if d["dataType"] == t and d["isCopyrighted"] and d["id"] not in exclude]
        bA, bB = [], []
        for i in full:
            r = M[i]["dataRequirements"].get(t)
            if r:
                bA.append((i, r["lowerBound"], r["upperBound"]))
        for i in half | source:
            r = M[i]["dataRequirements"].get(t)
            if r and (i, t) not in srcreq:
                bB.append((i, r["lowerBound"], r["upperBound"]))
        sF = sum(s for s, _ in fr)
        sC = sum(s for s, _ in cp)
        LA = sum(b[1] for b in bA)
        LB = sum(b[1] for b in bB)
        st = {"slackA": sF - LA, "slackT": sF + sC - LA - LB}
        # phase A
        pk = Packer(fr, rng, K=K, G0=G0)
        budget = max(0, int((sF - LA) * budget_frac)) if budget_frac is not None else 10**9
        wA = 0
        fA = 0
        for i, lb, ub in order_bins(bA, order, rng):
            if i in fset:
                continue
            if ub < lb:
                failed_bins.append((i, t)); fA += 1; fset.add(i)
                continue
            res = pk.fill(lb, ub, max(0, budget - wA))
            if res is None:
                if demote:
                    bB.append((i, lb, ub)); demoted.add(i)
                else:
                    failed_bins.append((i, t)); fA += 1; fset.add(i)
                continue
            wA += sum(s for s, _ in res) - lb
            for _, d in res:
                data_map[d] = i
        left = list(pk.pool)
        pk = Packer(cp + left, rng, K=K, G0=G0)
        budget = max(0, int((sF + sC - LA - LB - wA) * budget_frac)) if budget_frac is not None else 10**9
        wB = 0
        fB = 0
        for i, lb, ub in order_bins(bB, order, rng):
            if i in fset:
                continue
            if ub < lb:
                failed_bins.append((i, t)); fB += 1; fset.add(i)
                continue
            res = pk.fill(lb, ub, max(0, budget - wB))
            if res is None:
                failed_bins.append((i, t)); fB += 1; fset.add(i)
                continue
            wB += sum(s for s, _ in res) - lb
            for _, d in res:
                data_map[d] = i
        st.update(wA=wA, wB=wB, fA=fA, fB=fB, left=pk.total())
        stats[t] = st
        if verbose:
            print("  ", t, st, flush=True)
    failed = {i for i, _ in failed_bins}
    stats["demoted"] = len(demoted - failed)
    data_map = {d: m for d, m in data_map.items() if m not in failed}
    return data_map, failed, stats
