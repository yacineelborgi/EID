"""Exact packing of a plan (full / half models, no sources) with LNS repair.

Full bins (requirements of `full` models) must be filled with free datasets; the plan
leaves only a tiny free-data slack per type, so (almost) every bin must be filled at
exactly its lowerBound and (almost) every free dataset used. Initial pass: bins in
decreasing lowerBound, big items first, exact subset-sum closing preferring large items.
Unfilled bins are then repaired by ruin & recreate: release k random filled bins and
refill them together with the unfilled bin from the pool.

usage: python pack4.py datasets/4_precise.json plan.json [seed] [iters]
"""
import json
import random
import sys
import time

from sortedcontainers import SortedList

TYPES = "ntic"
BIG = 1 << 60


def dp_close(cands, g, hi):
    """cands: list of (size, id) sorted by decreasing size. Minimal-sum subset in [g, hi]
    (prefers large items on ties). Returns list of items or None."""
    if hi < 0:
        return None
    if g <= 0:
        return []
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
    return out


class Packer:
    def __init__(self, items, bins, rnd, K=200, G0=22000, keep=12000):
        # items: list of (size, id); bins: dict b -> (lb, ub)
        self.pool = SortedList(items)
        self.bins = bins
        self.fill = {}  # b -> list of items
        self.rnd = rnd
        self.K = K
        self.G0 = G0
        self.keep = keep

    def cands(self, hi):
        p = self.pool
        n = p.bisect_right((hi, BIG))
        K = self.K
        if n <= K:
            c = [p[j] for j in range(n)]
        else:
            h = K // 2
            c = [p[j] for j in range(n - h, n)]
            rest = n - h
            idx = sorted(self.rnd.sample(range(rest), K - h))
            c += [p[j] for j in idx]
        c.sort(key=lambda it: -it[0])
        return c

    def fill_bin(self, b, rand_big=0.0):
        lb, ub = self.bins[b]
        p = self.pool
        taken = []
        cur = 0
        while lb - cur > self.G0:
            lim = min(lb - cur - self.keep, ub - cur)
            i = p.bisect_right((lim, BIG))
            if i == 0:
                break
            if rand_big and self.rnd.random() < rand_big:
                i = max(1, i - self.rnd.randint(0, 30))
            it = p[i - 1]
            p.remove(it)
            taken.append(it)
            cur += it[0]
        g, hi = lb - cur, ub - cur
        res = dp_close(self.cands(hi), g, hi)
        if res is None:
            for it in taken:
                p.add(it)
            return False
        for it in res:
            p.remove(it)
        self.fill[b] = taken + res
        return True

    def release(self, b):
        for it in self.fill.pop(b):
            self.pool.add(it)

    def waste(self):
        return sum(sum(s for s, _ in its) - self.bins[b][0] for b, its in self.fill.items())

    def initial(self, order):
        failed = []
        for b in order:
            if not self.fill_bin(b):
                failed.append(b)
        return failed

    def repair(self, failed, iters=2000000, kmin=1, kmax=8, tlimit=120, log=False, budget=None):
        rnd = self.rnd
        unfilled = list(failed)
        t0 = time.time()
        it = 0
        best = len(unfilled)
        for it in range(iters):
            if not unfilled or time.time() - t0 > tlimit:
                break
            u = rnd.choice(unfilled)
            lbu, ubu = self.bins[u]
            k = rnd.randint(kmin, kmax)
            filled = list(self.fill.keys())
            chosen = set()
            if rnd.random() < 0.7:
                # bins that own items small enough to be useful for u
                owner = {}
                for b in filled:
                    for x in self.fill[b]:
                        if x[0] <= ubu:
                            owner.setdefault(b, 0)
                            owner[b] += 1
                cand = list(owner)
                if cand:
                    chosen = set(rnd.sample(cand, min(k, len(cand))))
            while len(chosen) < k:
                chosen.add(rnd.choice(filled))
            chosen = list(chosen)
            old = {b: self.fill[b] for b in chosen}
            for b in chosen:
                self.release(b)
            order = [u] + chosen
            mode = rnd.random()
            if mode < 0.4:
                order.sort(key=lambda b: -self.bins[b][0])
            elif mode < 0.7:
                rnd.shuffle(order)
            nfail = []
            for b in order:
                if not self.fill_bin(b, rand_big=0.3):
                    nfail.append(b)
                    if len(nfail) > 1:
                        break
            wok = budget is None or self.waste() <= budget
            if wok and (not nfail or (len(nfail) == 1 and nfail[0] != u and rnd.random() < 0.3)):
                unfilled.remove(u)
                unfilled += nfail
                if log and not nfail:
                    print(f"    repaired {u} at iter {it} (k={k}), left {len(unfilled)}", flush=True)
            else:
                for b in order:
                    if b in self.fill and (b == u or b in old):
                        self.release(b)
                for b in chosen:
                    for x in old[b]:
                        self.pool.remove(x)
                    self.fill[b] = old[b]
        return unfilled, it


def load_inst(path):
    with open(path) as f:
        txt = f.read()
    return txt, json.loads(txt)


def pack_plan(inst, plan, seed=0, tlimit=120, log=True, tries=3):
    M, D = inst["models"], inst["datasets"]
    full = set(plan["full"])
    half = set(plan["half"])
    rnd = random.Random(seed)
    data_map = {}
    failed_models = set()
    packers = {}
    for t in TYPES:
        items = [(d["size"], d["id"]) for d in D if d["dataType"] == t and not d["isCopyrighted"]]
        bins = {}
        for i in full:
            r = M[i]["dataRequirements"].get(t)
            if r:
                bins[i] = (r["lowerBound"], r["upperBound"])
        slack = sum(s for s, _ in items) - sum(lb for lb, _ in bins.values())
        bestP = None
        for attempt in range(tries):
            P = Packer(items, bins, rnd)
            order = sorted(bins, key=lambda b: -bins[b][0])
            t0 = time.time()
            failed = P.initial(order)
            if log:
                print(f"  type {t} try {attempt}: bins {len(bins)} slack {slack} "
                      f"initial failed {len(failed)} waste {P.waste()} pool {len(P.pool)} ({time.time() - t0:.1f}s)", flush=True)
            if failed:
                failed, its = P.repair(failed, tlimit=tlimit, log=False, budget=slack)
                if log:
                    print(f"    after repair: failed {len(failed)} iters {its} waste {P.waste()} pool {len(P.pool)} "
                          f"poolsum {sum(s for s, _ in P.pool)} ({time.time() - t0:.1f}s)", flush=True)
            if bestP is None or len(failed) < len(bestP[1]):
                bestP = (P, failed)
            if not failed:
                break
        P, failed = bestP
        failed_models |= set(failed)
        packers[t] = P
    # models failed on some type: release their free data everywhere, make them half
    for t in TYPES:
        P = packers[t]
        for b in list(P.fill):
            if b in failed_models:
                P.release(b)
        for b, its in P.fill.items():
            for s, d in its:
                data_map[d] = b
    full_ok = full - failed_models
    half_all = half | failed_models
    # half bins: copyrighted first, then leftover free
    copy = {t: SortedList((d["size"], d["id"]) for d in D if d["dataType"] == t and d["isCopyrighted"]) for t in TYPES}
    half_ok = set()
    half_failed = set()
    for i in sorted(half_all, key=lambda i: -M[i]["value"]):
        tk = []
        ok = True
        for t, r in M[i]["dataRequirements"].items():
            got = None
            for pool in (copy[t], packers[t].pool):
                Q = Packer([], {0: (r["lowerBound"], r["upperBound"])}, rnd)
                Q.pool = pool
                if Q.fill_bin(0):
                    got = (pool, Q.fill[0])
                    break
            if got is None:
                ok = False
                break
            tk.append(got)
        if ok:
            half_ok.add(i)
            for pool, its in tk:
                for s, d in its:
                    data_map[d] = i
        else:
            half_failed.add(i)
            for pool, its in tk:
                for it in its:
                    pool.add(it)
    return data_map, full_ok, half_ok, failed_models, half_failed, packers, copy


def greedy_more(inst, data_map, used, packers, copy, rnd):
    """Add unused models (half mode with copyrighted data, or full if free data allows)."""
    M = inst["models"]
    cap = inst["energyCap"]
    energy = sum(M[i]["energyCost"] for i in used)
    cand = [i for i in range(len(M)) if i not in used and M[i]["value"] > 0
            and all(r["upperBound"] >= r["lowerBound"] for r in M[i]["dataRequirements"].values())]
    cand.sort(key=lambda i: -M[i]["value"] / max(M[i]["energyCost"], 1))
    added = []
    for i in cand:
        m = M[i]
        if energy + m["energyCost"] > cap:
            continue
        for mode in ("full", "half"):
            tk = []
            ok = True
            for t, r in m["dataRequirements"].items():
                pools = [packers[t].pool] if mode == "full" else [copy[t], packers[t].pool]
                got = None
                for pool in pools:
                    Q = Packer([], {0: (r["lowerBound"], r["upperBound"])}, rnd)
                    Q.pool = pool
                    if Q.fill_bin(0):
                        got = (pool, Q.fill[0])
                        break
                if got is None:
                    ok = False
                    break
                tk.append(got)
            if ok:
                for pool, its in tk:
                    for s, d in its:
                        data_map[d] = i
                energy += m["energyCost"]
                used.add(i)
                added.append((i, mode))
                break
            for pool, its in tk:
                for it in its:
                    pool.add(it)
    return added


if __name__ == "__main__":
    sys.path.insert(0, ".")
    from solve import strict_ok
    from test_solution import getSolutionScore
    path, pf = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tlimit = float(sys.argv[4]) if len(sys.argv) > 4 else 120
    out = sys.argv[5] if len(sys.argv) > 5 else None
    txt, inst = load_inst(path)
    plan = json.load(open(pf))
    M = inst["models"]
    print("plan obj", plan.get("obj"), "full", len(plan["full"]), "half", len(plan["half"]), flush=True)
    dm, fo, ho, fm, hf, packers, copy = pack_plan(inst, plan, seed=seed, tlimit=tlimit)
    pre = sum(M[i]["value"] for i in fo) + sum(M[i]["value"] // 2 for i in ho)
    print("failed full", len(fm), "failed half", len(hf), "pre", pre, flush=True)
    used = fo | ho
    added = greedy_more(inst, dm, used, packers, copy, random.Random(seed))
    sol = {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": []}
    sc, ok, msg = getSolutionScore(json.dumps(sol), txt, show_details=False)
    print("added", added, "score", sc, ok, msg, "strict", strict_ok(inst, sol), flush=True)
    if out and ok and strict_ok(inst, sol):
        try:
            o = json.load(open(out))
            osc, ook, _ = getSolutionScore(json.dumps(o), txt, show_details=False)
        except Exception:
            osc, ook = -1, False
        if not ook or sc > osc:
            json.dump(sol, open(out, "w"))
            print("saved", out, sc, "previous", osc if ook else None)
