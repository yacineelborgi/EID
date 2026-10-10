"""Plan -> legitimate solution for 7_big using packer.py; greedy fill of leftover energy; check & save.
usage: build7.py plan.json [seeds] [G0]"""
import sys, json, time, os, random
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
sys.path.insert(0, "/home/user/EID")
from data7 import load
from packer import pack_type, Pool, fill_bin
OUT = "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_max/7_big_submission.json"
LOCAL = "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7/best7.json"
CUR = 5822389
TY = "ntic"
d = load()
v, e, lb, ub, has, ds, dt, dc = d["v"], d["e"], d["lb"], d["ub"], d["has"], d["ds"], d["dt"], d["dc"]
cap = int(d["cap"]); nM = len(v)
badt = (ub < lb) & has; bad = badt.any(1); mono = has.sum(1) == 1

def build(plan, seed=0, G0=12000, verbose=True):
    rng = random.Random(seed)
    full = set(plan["full"]); half = set(plan["half"]); src = set(plan["source"])
    dmap = {}  # dataset -> model
    failed = set()
    leftover_free = {}
    stats = {}
    for t in range(4):
        items = [(int(ds[k]), int(k)) for k in np.nonzero((dt == t) & (dc == 0))[0]]
        bins = [(i, int(lb[i, t]), int(ub[i, t])) for i in full if has[i, t]]
        out, fl, w, pool = pack_type(bins, items, rng, G0=G0)
        for m, its in out.items():
            for _, k in its: dmap[k] = m
        failed |= {m for m, _, _ in fl}
        stats[t] = dict(waste=w, failed=len(fl), left=pool.total(), nleft=len(pool.sl))
        leftover_free[t] = list(pool.sl)
    if failed:
        # release data of failed full models
        rel = [k for k, m in dmap.items() if m in failed]
        for k in rel:
            leftover_free[int(dt[k])].append((int(ds[k]), k)); del dmap[k]
    full2 = full - failed
    # Phase B: half (non-bad reqs) + sources, copyrighted first then free leftovers
    pools = {}
    failedB = set()
    for t in range(4):
        cp = [(int(ds[k]), int(k)) for k in np.nonzero((dt == t) & (dc == 1))[0]]
        pool = Pool(cp)
        bins = [(i, int(lb[i, t]), int(ub[i, t])) for i in (half | src) if has[i, t] and not badt[i, t]]
        bins.sort(key=lambda b: b[1])
        for m, l, u in bins:
            res = fill_bin(pool, l, u, rng, 10**9, G0=G0)
            if res is None:
                failedB.add(m); continue
            for _, k in res: dmap[k] = m
        pools[t] = (pool, Pool(leftover_free[t]))
    if failedB:
        for k in [k for k, m in dmap.items() if m in failedB]:
            del dmap[k]
    half2 = half - failedB; src2 = src - failedB
    # match sources to bad reqs
    mm = []
    by_t = {t: sorted([i for i in src2 if has[i, t]], key=lambda i: e[i]) for t in range(4)}
    need = {t: sorted([i for i in half2 if badt[i, t]], key=lambda i: -v[i]) for t in range(4)}
    unmatched = set()
    for t in range(4):
        for k, i in enumerate(need[t]):
            if k < len(by_t[t]): mm.append([by_t[t][k], i])
            else: unmatched.add(i)
    if unmatched:
        half2 -= unmatched
        mm = [p for p in mm if p[1] in half2]
        for k in [k for k, m in dmap.items() if m in unmatched]:
            del dmap[k]
    used_src = {s for s, _ in mm}
    unused_src = src2 - used_src
    for k in [k for k, m in dmap.items() if m in unused_src]:
        del dmap[k]
    used = full2 | half2 | used_src
    energy = int(sum(int(e[i]) for i in used))
    # free leftovers back: datasets not in dmap
    info = dict(stats=stats, failedA=len(failed), failedB=len(failedB), unmatched=len(unmatched),
                unused_src=len(unused_src), energy_slack=cap - energy)
    return dmap, mm, full2, half2, used_src, energy, info

def greedy_fill(dmap, mm, full2, half2, used_src, energy, rng):
    """add unused models (half with copyrighted data, or full with free leftovers) within energy slack."""
    usedd = set(dmap)
    pf = {t: Pool([(int(ds[k]), int(k)) for k in np.nonzero((dt == t) & (dc == 0))[0] if k not in usedd]) for t in range(4)}
    pc = {t: Pool([(int(ds[k]), int(k)) for k in np.nonzero((dt == t) & (dc == 1))[0] if k not in usedd]) for t in range(4)}
    used = full2 | half2 | used_src
    slack = cap - energy
    cand = [i for i in range(nM) if i not in used and not bad[i] and e[i] <= slack and v[i] // 2 > 0]
    cand.sort(key=lambda i: -v[i] / max(int(e[i]), 1))
    added = 0; gain = 0
    for i in cand:
        if e[i] > slack: continue
        ts = np.nonzero(has[i])[0]
        for mode in ("full", "half"):
            taken = []; ok = True
            for t in ts:
                p = pf[t] if mode == "full" else pc[t]
                res = fill_bin(p, int(lb[i, t]), int(ub[i, t]), rng, 10**9)
                if res is None: ok = False; break
                taken.append((p, res))
            if ok:
                for p, res in taken:
                    for _, k in res: dmap[k] = i
                (full2 if mode == "full" else half2).add(i)
                slack -= int(e[i]); added += 1; gain += int(v[i]) if mode == "full" else int(v[i]) // 2
                break
            for p, res in taken:
                for it in res: p.add(it)
    return added, gain, cap - slack

if __name__ == "__main__":
    from solve import score_of, strict_ok
    txt = open("/home/user/EID/datasets/7_big.json").read()
    inst = json.loads(txt)
    plan = json.load(open(sys.argv[1]))
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    G0 = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
    best = CUR
    for f in (OUT, LOCAL):
        if os.path.exists(f):
            s, ok, _ = score_of(txt, json.load(open(f)))
            if ok: best = max(best, s)
    pv = int(v[plan["full"]].sum() + (v[plan["half"]] // 2).sum())
    pe = int(sum(int(e[i]) for i in plan["full"] + plan["half"] + plan["source"]))
    print("best so far", best, "plan value", pv, "plan energy over cap", pe - cap, flush=True)
    for seed in seeds:
        t0 = time.time()
        dmap, mm, full2, half2, used_src, energy, info = build(plan, seed, G0)
        pre = int(v[list(full2)].sum() + (v[list(half2)] // 2).sum())
        print(f" seed {seed}: pre {pre} {info}", flush=True)
        if energy > cap:
            print("  energy over cap by", energy - cap, flush=True)
            continue
        added, gain, energy = greedy_fill(dmap, mm, full2, half2, used_src, energy, random.Random(seed))
        sol = {"dataMappings": [[int(k), int(m)] for k, m in dmap.items()], "modelMappings": [[int(a), int(b)] for a, b in mm]}
        sc, ok, msg = score_of(txt, sol)
        ok2 = strict_ok(inst, sol)
        print(f"  added {added} gain {gain} score {sc} {ok} {ok2} {msg} {time.time()-t0:.0f}s", flush=True)
        if ok and ok2 and sc > best:
            best = sc
            json.dump(sol, open(LOCAL, "w")); json.dump(sol, open(OUT, "w"))
            print("  wrote", sc, flush=True)
