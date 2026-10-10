"""Exact packer for 7_big plans (pack5 approach + MIP end-game + lnsfix-style rescue).

Phase A per type (full bins, free items only):
  1. tiny bins (lb <= theta): one item each, matroid greedy respecting ub (min total size);
  2. other bins except the last `nfinal`: exact fill (sum == lb) with big items + subset-sum DP;
  3. last `nfinal` bins: assignment MIP over all remaining items (windows [lb,ub]);
  4. failures: rescue MIP with donor bins.
Phase B per type (half + source bins): copyrighted items + free leftovers, any waste within ub.
usage: pack7b.py plan.json [seeds] [order] [theta] [nfinal]
"""
import sys, json, time, os, random
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/e5")
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6")
sys.path.insert(0, "/home/user/EID")
from sortedcontainers import SortedList
from exact_assign import Packer
from pack5 import tiny_match
from lnsfix import solve_assign
from data7 import load

OUT = "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_max/7_big_submission.json"
LOCAL = "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7/best7.json"
CUR = 5822389
d = load()
v, e, lb, ub, has, ds, dt, dc = d["v"], d["e"], d["lb"], d["ub"], d["has"], d["ds"], d["dt"], d["dc"]
cap = int(d["cap"]); nM = len(v)
badt = (ub < lb) & has; bad = badt.any(1); mono = has.sum(1) == 1


def final_mip(bins, pool, tl=60.0, log=print):
    """bins: list (i, lb, ub); pool: SortedList (size,id). Assign all bins from pool items via MIP."""
    items = list(pool)
    sizes = [s for s, _ in items]
    res = solve_assign([(l, u) for _, l, u in bins], sizes, lambda b, j: True, tl=tl)
    if res is None:
        return None
    out = {}
    for b, js in enumerate(res):
        out[bins[b][0]] = [items[j] for j in js]
    for its in out.values():
        for it in its:
            pool.remove(it)
    return out


def rescue_bin(fb, res, pool, rng, ndonor=25, tl=10.0, tries=4):
    """fb=(i,lb,ub) failed bin; res dict i->items (filled bins of the same pool). Re-solve fb + donors."""
    keys = list(res.keys())
    for _ in range(tries):
        # donors: bins with similar size range, random
        donors = rng.sample(keys, min(ndonor, len(keys)))
        bins = [fb] + [(m, int(lb[m, T]), int(ub[m, T])) for m in donors]
        items = list(pool) + [it for m in donors for it in res[m]]
        sizes = [s for s, _ in items]
        sol = solve_assign([(l, u) for _, l, u in bins], sizes, lambda b, j: True, tl=tl)
        if sol is None:
            continue
        for m in donors:
            for it in res[m]:
                pool.add(it)
        for b, js in enumerate(sol):
            m = bins[b][0]
            res[m] = [items[j] for j in js]
            for j in js:
                pool.remove(items[j])
        return True
    return False


T = 0


def pack_full(t, bins, items, rng, theta=300, nfinal=40, order="lb_desc", K=300, G0=20000, log=print):
    global T
    T = t
    pool = SortedList(items)
    tiny = [b for b in bins if b[1] <= theta]
    rest = [b for b in bins if b[1] > theta]
    asg, failed = tiny_match(tiny, pool)
    res = {i: [it] for i, it in asg.items()}
    if order == "lb_desc":
        rest.sort(key=lambda b: -b[1])
    elif order == "lb_asc":
        rest.sort(key=lambda b: b[1])
    elif order == "random":
        rng.shuffle(rest)
    elif order == "mix":  # big first, then random, keep wide windows for the end
        rest.sort(key=lambda b: -b[1])
    # final bins: the widest windows among the rest (most flexible)
    if nfinal > 0:
        fin_ids = {b[0] for b in sorted(rest, key=lambda b: -(b[2] - b[1]))[:nfinal]}
    else:
        fin_ids = set()
    fin = [b for b in rest if b[0] in fin_ids]
    rest = [b for b in rest if b[0] not in fin_ids]
    pk = Packer([], rng, K=K, G0=G0)
    pk.pool = pool
    for i, l, u in rest:
        r = pk.fill(l, u, 0)
        if r is None:
            r = pk.fill(l, u, 30)
            if r is not None:
                log(f"   bin {i} lb {l} filled with waste {sum(s for s, _ in r) - l}")
        if r is None:
            failed.append((i, l, u)); continue
        res[i] = r
    left = sum(s for s, _ in pool)
    log(f"   t{t}: tiny {len(tiny)} rest {len(rest)} fin {len(fin)} failed-pre {len(failed)} "
        f"pool {len(pool)} left {left} finlb {sum(b[1] for b in fin)}")
    if fin:
        fin.sort(key=lambda b: (b[2] - b[1]))  # widest last
        slack = sum(s for s, _ in pool) - sum(b[1] for b in fin)
        for k, (i, l, u) in enumerate(fin):
            last = (k == len(fin) - 1)
            pk.K = 2000
            rr = pk.fill(l, u, (u - l) if last else 0)
            if rr is None and not last:
                rr = pk.fill(l, u, max(0, min(u - l, slack // 4)))
            if rr is None:
                failed.append((i, l, u)); continue
            res[i] = rr
            slack -= sum(s for s, _ in rr) - l
        pk.K = K
    # rescue failures
    still = []
    for fb in failed:
        if rescue_bin(fb, res, pool, rng):
            log(f"   rescued bin {fb}")
        else:
            still.append(fb)
    waste = sum(sum(s for s, _ in res[i]) - l for i, l, u in bins if i in res)
    return res, still, waste, list(pool)


def pack_half(t, bins, items, rng, K=300, G0=20000):
    pool = SortedList(items)
    pk = Packer([], rng, K=K, G0=G0)
    pk.pool = pool
    res = {}; failed = []
    for i, l, u in sorted(bins, key=lambda b: (b[2] - b[1], b[1])):
        r = pk.fill(l, u, u - l)
        if r is None:
            failed.append((i, l, u)); continue
        res[i] = r
    return res, failed


def build(plan, seed=0, order="lb_desc", theta=300, nfinal=40, log=print):
    rng = random.Random(seed)
    full = set(plan["full"]); half = set(plan["half"]); src = set(plan["source"])
    dmap = {}
    failed = set()
    leftover = {}
    for t in range(4):
        items = [(int(ds[k]), int(k)) for k in np.nonzero((dt == t) & (dc == 0))[0]]
        bins = [(i, int(lb[i, t]), int(ub[i, t])) for i in full if has[i, t]]
        res, fl, w, left = pack_full(t, bins, items, rng, theta, nfinal, order, log=log)
        log(f"  type {t}: full bins {len(bins)} failed {len(fl)} waste {w} leftover {sum(s for s, _ in left)}")
        for m, its in res.items():
            for _, k in its:
                dmap[k] = m
        failed |= {b[0] for b in fl}
        leftover[t] = left
    if failed:
        for k in [k for k, m in dmap.items() if m in failed]:
            leftover[int(dt[k])].append((int(ds[k]), k)); del dmap[k]
    full2 = full - failed
    failedB = set()
    for t in range(4):
        cp = [(int(ds[k]), int(k)) for k in np.nonzero((dt == t) & (dc == 1))[0]]
        bins = [(i, int(lb[i, t]), int(ub[i, t])) for i in (half | src) if has[i, t] and not badt[i, t]]
        res, fl = pack_half(t, bins, cp + leftover[t], rng)
        for m, its in res.items():
            for _, k in its:
                dmap[k] = m
        failedB |= {b[0] for b in fl}
    if failedB:
        log(f"  phase B failed {len(failedB)}")
        for k in [k for k, m in dmap.items() if m in failedB]:
            del dmap[k]
    half2 = half - failedB; src2 = src - failedB
    mm = []
    by_t = {t: sorted([i for i in src2 if has[i, t]], key=lambda i: e[i]) for t in range(4)}
    need = {t: sorted([i for i in half2 if badt[i, t]], key=lambda i: -v[i]) for t in range(4)}
    unmatched = set()
    for t in range(4):
        for k, i in enumerate(need[t]):
            if k < len(by_t[t]): mm.append([int(by_t[t][k]), int(i)])
            else: unmatched.add(i)
    if unmatched:
        half2 -= unmatched
        mm = [p for p in mm if p[1] in half2]
        for k in [k for k, m in dmap.items() if m in unmatched]:
            del dmap[k]
    used_src = {s for s, _ in mm}
    for k in [k for k, m in dmap.items() if m in (src2 - used_src)]:
        del dmap[k]
    energy = int(sum(int(e[i]) for i in full2 | half2 | used_src))
    return dmap, mm, full2, half2, used_src, energy, failed, failedB


if __name__ == "__main__":
    from solve import score_of, strict_ok
    from build7 import greedy_fill
    txt = open("/home/user/EID/datasets/7_big.json").read()
    inst = json.loads(txt)
    plan = json.load(open(sys.argv[1]))
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    order = sys.argv[3] if len(sys.argv) > 3 else "lb_desc"
    theta = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    nfinal = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    best = CUR
    for f in (OUT, LOCAL):
        if os.path.exists(f):
            s, ok, _ = score_of(txt, json.load(open(f)))
            if ok and strict_ok(inst, json.load(open(f))): best = max(best, s)
    pv = int(v[plan["full"]].sum() + (v[plan["half"]] // 2).sum())
    pe = int(sum(int(e[i]) for i in plan["full"] + plan["half"] + plan["source"]))
    print("best so far", best, "plan value", pv, "plan energy - cap", pe - cap, flush=True)
    for seed in seeds:
        t0 = time.time()
        dmap, mm, full2, half2, used_src, energy, fA, fB = build(plan, seed, order, theta, nfinal,
                                                                 log=lambda s: print(s, flush=True))
        pre = int(v[list(full2)].sum() + (v[list(half2)] // 2).sum())
        print(f" seed {seed}: pre {pre} failedA {sorted(fA)} failedB {sorted(fB)} energy-cap {energy-cap}", flush=True)
        if energy > cap:
            continue
        added, gain, energy = greedy_fill(dmap, mm, full2, half2, used_src, energy, random.Random(seed))
        sol = {"dataMappings": [[int(k), int(m)] for k, m in dmap.items()], "modelMappings": [[int(a), int(b)] for a, b in mm]}
        sc, ok, msg = score_of(txt, sol)
        ok2 = strict_ok(inst, sol)
        E = sum(int(inst["models"][i]["energyCost"]) for i in set(m for _, m in sol["dataMappings"]) | {s for s, _ in mm} | {t for _, t in mm})
        print(f"  added {added} gain {gain} score {sc} {ok} {ok2} {msg} intE-cap {E - cap} {time.time()-t0:.0f}s", flush=True)
        json.dump(sol, open(f"/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7/sol_{os.path.basename(sys.argv[1])[:-5]}_{seed}.json", "w"))
        if ok and ok2 and E <= cap and sc > best:
            best = sc
            json.dump(sol, open(LOCAL, "w")); json.dump(sol, open(OUT, "w"))
            print("  wrote", sc, flush=True)
