"""Driver v2: small-item aware MIP plan -> assignment restarts -> optimal source matching
-> greedy fill (with spare sources) -> residual MIP rounds.

usage: python solve2.py datasets/6_shortage.json plans/<plan>.json [restarts] [resid_rounds] [resid_time]
"""
import json
import os
import random
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from assign import TYPES, Pool, fill_bin
from solve import score_of, strict_ok


def assign2(inst, plan, rnd, noise=0.0, exclude=(), order="lb_asc"):
    M, D = inst["models"], inst["datasets"]
    full = set(plan["full"])
    half = set(plan["half"])
    source = set(plan["source"])
    srcreq = {(i, t) for i, t in plan["srcreq"]}
    exclude = set(exclude)
    free = {t: Pool([]) for t in TYPES}
    copy = {t: Pool([]) for t in TYPES}
    for d in D:
        if d["id"] in exclude:
            continue
        (copy if d["isCopyrighted"] else free)[d["dataType"]].put((d["size"], d["id"]))
    data_map = {}
    failed = set()
    types = list(TYPES)
    for t in types:
        bins_free, bins_any = [], []
        for i in full:
            r = M[i]["dataRequirements"].get(t)
            if r:
                bins_free.append((i, r["lowerBound"], r["upperBound"]))
        for i in half | source:
            r = M[i]["dataRequirements"].get(t)
            if r and (i, t) not in srcreq:
                bins_any.append((i, r["lowerBound"], r["upperBound"]))
        if order == "lb_asc":
            key = lambda b: b[1] * (1 + noise * rnd.random())
        else:
            key = lambda b: -b[1] * (1 + noise * rnd.random())
        bins_free.sort(key=key)
        bins_any.sort(key=key)
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
    return data_map, failed, free, copy


def _sums(items):
    """dict sum -> list of items reaching it (small item sets only)."""
    r = {0: []}
    for it in items:
        for v, l in list(r.items()):
            nv = v + it[0]
            if nv not in r:
                r[nv] = l + [it]
    return r


def repair(inst, plan, data_map, failed, max_pool=12):
    """Try to rescue failed models: release their datasets, then fill each of their bins from
    the leftover pool, using 1-swaps with already filled bins (replace item a of bin X by a
    leftover item b, X stays in its window) when the pool alone cannot close the bin."""
    M, D = inst["models"], inst["datasets"]
    full = set(plan["full"])
    srcreq = {(i, t) for i, t in plan["srcreq"]}
    for d in [d for d, m in data_map.items() if m in failed]:
        del data_map[d]
    content = {}
    for d, m in data_map.items():
        t = D[d]["dataType"]
        content.setdefault((m, t), []).append((D[d]["size"], d))
    pool = {t: [] for t in TYPES}
    for d in D:
        if d["id"] not in data_map:
            pool[d["dataType"]].append((d["size"], d["id"]))
    isfree = lambda it: not D[it[1]]["isCopyrighted"]
    rescued = set()
    order = sorted(failed, key=lambda i: (i not in plan["source"], -M[i]["value"]))
    for i in order:
        reqs = [(t, r) for t, r in M[i]["dataRequirements"].items() if (i, t) not in srcreq]
        if any(r["upperBound"] < r["lowerBound"] for _, r in reqs):
            continue
        done = []  # (t, items, swaps)
        ok = True
        for t, r in reqs:
            lb, ub = r["lowerBound"], r["upperBound"]
            fr = i in full
            avail = [it for it in pool[t] if (isfree(it) or not fr)]
            avail.sort(reverse=True)
            got = None
            # direct: subset of pool (small pool -> exact enumeration on largest items)
            cand = avail[:max_pool]
            S = _sums(cand)
            best = None
            for v, l in S.items():
                if lb <= v <= ub and (best is None or v < best[0]):
                    best = (v, l)
            if best:
                got = (best[1], None)
            else:
                # 1-swap: X gives item a (taken by bin i), receives leftover b
                import bisect
                rs = {}
                for b in cand:
                    d = _sums([c for c in cand if c != b])
                    ks = sorted(d)
                    rs[b] = (ks, d)
                for (x, tx), its in content.items():
                    if tx != t or x == i:
                        continue
                    rx = M[x]["dataRequirements"][t]
                    fx = sum(it[0] for it in its)
                    xfree = x in full
                    for a in its:
                        if fr and not isfree(a):
                            continue
                        for b in cand:
                            if xfree and not isfree(b):
                                continue
                            nf = fx - a[0] + b[0]
                            if not (rx["lowerBound"] <= nf <= rx["upperBound"]):
                                continue
                            ks, d = rs[b]
                            j = bisect.bisect_left(ks, lb - a[0])
                            if j < len(ks) and ks[j] <= ub - a[0]:
                                got = (d[ks[j]] + [a], (x, t, a, b))
                                break
                        if got:
                            break
                    if got:
                        break
            if got is None:
                ok = False
                break
            items, sw = got
            if sw:
                x, tx, a, b = sw
                content[(x, tx)].remove(a); content[(x, tx)].append(b)
                pool[t].remove(b); pool[t].append(a)
                data_map[b[1]] = x
            for it in items:
                pool[t].remove(it)
                data_map[it[1]] = i
            content[(i, t)] = list(items)
            done.append((t, items))
        if ok:
            rescued.add(i)
        else:
            for t, items in done:
                for it in items:
                    pool[t].append(it)
                    del data_map[it[1]]
                del content[(i, t)]
    return failed - rescued


def choose_targets(inst, cands, avail):
    """cands: list of (model, types needing a source). avail: {t: count}.
    Maximise sum value//2 with per-type source counts. Returns chosen set."""
    M = inst["models"]
    if not cands:
        return set()
    n = len(cands)
    c = -np.array([M[i]["value"] // 2 for i, _ in cands], float)
    A = np.zeros((4, n))
    for k, (i, ts) in enumerate(cands):
        for t in ts:
            A[TYPES.index(t), k] += 1
    hi = np.array([avail.get(t, 0) for t in TYPES], float)
    res = milp(c, constraints=LinearConstraint(A, -np.inf, hi), integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"time_limit": 30})
    x = np.round(res.x).astype(int)
    return {cands[k][0] for k in range(n) if x[k]}


def finalize2(inst, plan, data_map, failed):
    M = inst["models"]
    full = set(plan["full"]) - failed
    half = set(plan["half"]) - failed
    sources = [i for i in plan["source"] if i not in failed]
    by_type = {t: [] for t in TYPES}
    for i in sources:
        by_type[next(iter(M[i]["dataRequirements"]))].append(i)
    need = {}
    for i, t in plan["srcreq"]:
        if i in half:
            need.setdefault(i, []).append(t)
    # half models without sourced requirement (copyright data only) are kept as is
    chosen = choose_targets(inst, list(need.items()), {t: len(v) for t, v in by_type.items()})
    half = {i for i in half if i not in need or i in chosen}
    model_map = []
    ptr = {t: 0 for t in TYPES}
    for i in sorted(chosen):
        for t in need[i]:
            model_map.append([by_type[t][ptr[t]], i])
            ptr[t] += 1
    spare = {t: by_type[t][ptr[t]:] for t in TYPES}  # filled, unused sources
    used_src = {s for s, _ in model_map}
    keep = full | half | used_src | {s for t in TYPES for s in spare[t]}
    data_map = {d: m for d, m in data_map.items() if m in keep}
    return data_map, model_map, full, half, used_src, spare


def greedy_add2(inst, data_map, model_map, full, half, used_src, spare, rnd=None, noise=0.0, priority=(),
                new_src=True, max_src_tries=100):
    """Add unused models with leftover energy, datasets, spare (already filled) sources and,
    if new_src, freshly created sources (unused mono models filled from leftover data).
    Spare sources still unused at the end are removed (their datasets released)."""
    M, D = inst["models"], inst["datasets"]
    cap = inst["energyCap"]
    spare = {t: list(v) for t, v in spare.items()}
    spare_set = {s for v in spare.values() for s in v}
    energy = sum(M[i]["energyCost"] for i in full | half | used_src | spare_set)
    free = {t: Pool([]) for t in TYPES}
    copy = {t: Pool([]) for t in TYPES}
    for d in D:
        if d["id"] not in data_map:
            (copy if d["isCopyrighted"] else free)[d["dataType"]].put((d["size"], d["id"]))
    used = full | half | used_src | spare_set
    mono = {t: [] for t in TYPES}
    for i, m in enumerate(M):
        if len(m["dataRequirements"]) == 1 and i not in used:
            t, r = next(iter(m["dataRequirements"].items()))
            if r["upperBound"] >= r["lowerBound"]:
                mono[t].append(i)
    for t in TYPES:
        mono[t].sort(key=lambda i: M[i]["energyCost"])
    cand = [i for i in range(len(M)) if i not in used and M[i]["value"] > 0]
    key = {i: -M[i]["value"] / max(M[i]["energyCost"], 1) * (1 + noise * (rnd.random() if rnd else 0)) for i in cand}
    pr = set(priority)
    cand.sort(key=lambda i: (i not in pr, key[i]))
    added = 0
    for i in cand:
        if i in used:
            continue
        m = M[i]
        reqs = m["dataRequirements"]
        if energy + m["energyCost"] > cap:
            continue
        for mode in ("full", "half"):
            taken, srcs, made = [], [], []
            e_extra = 0
            ok = True
            for t, r in reqs.items():
                if r["upperBound"] >= r["lowerBound"]:
                    pools = [free[t]] if mode == "full" else [copy[t], free[t]]
                    res = fill_bin(r["lowerBound"], r["upperBound"], pools)
                    if res is not None:
                        taken += res
                        continue
                if mode == "half" and spare[t]:
                    srcs.append((t, spare[t].pop()))
                    continue
                if mode == "half" and new_src:
                    got = None
                    tries = 0
                    for s in mono[t]:
                        if s in used or s == i or s in {x for _, x, _ in made}:
                            continue
                        if energy + m["energyCost"] + e_extra + M[s]["energyCost"] > cap:
                            break  # sorted by energy
                        tries += 1
                        if tries > max_src_tries:
                            break
                        rs = M[s]["dataRequirements"][t]
                        res = fill_bin(rs["lowerBound"], rs["upperBound"], [copy[t], free[t]])
                        if res is not None:
                            got = (t, s, res)
                            break
                    if got:
                        made.append(got)
                        e_extra += M[got[1]]["energyCost"]
                        continue
                ok = False
                break
            if ok:
                for _, it in taken:
                    data_map[it[1]] = i
                for t, s in srcs:
                    model_map.append([s, i])
                    used_src.add(s)
                for t, s, res in made:
                    for _, it in res:
                        data_map[it[1]] = s
                    model_map.append([s, i])
                    used_src.add(s)
                    used.add(s)
                (full if mode == "full" else half).add(i)
                used.add(i)
                energy += m["energyCost"] + e_extra
                added += 1
                break
            for p, it in taken:
                p.put(it)
            for t, s in srcs:
                spare[t].append(s)
            for t, s, res in made:
                for p, it in res:
                    p.put(it)
    # drop spare sources that remain unused
    left = {s for v in spare.values() for s in v}
    for d in [d for d, m in data_map.items() if m in left]:
        del data_map[d]
    return added


def build_solution(inst, plan, rnd, noise, order="lb_asc", exclude=(), base=None):
    dm, failed, _, _ = assign2(inst, plan, rnd, noise, exclude=exclude, order=order)
    nf0 = len(failed)
    for _ in range(4):
        nb = len(failed)
        failed = repair(inst, plan, dm, failed)
        if len(failed) == nb:
            break
    print(f"    repair: {nf0} -> {len(failed)} failed", flush=True)
    dm, mm, full, half, us, spare = finalize2(inst, plan, dm, failed)
    if base is not None:
        bdm, bmm, bfull, bhalf, bus = base
        dm.update(bdm); mm = bmm + mm
        full |= bfull; half |= bhalf; us |= bus
    pr = [i for i in plan["half"] + plan["full"] if i not in full and i not in half]
    added = greedy_add2(inst, dm, mm, full, half, us, spare, rnd, noise, priority=pr)
    sol = {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": mm}
    return sol, len(failed), added


def classify(inst, sol):
    M, D = inst["models"], inst["datasets"]
    dm = {d: m for d, m in sol["dataMappings"]}
    mm = sol["modelMappings"]
    us = {s for s, _ in mm}
    tg = {t for _, t in mm}
    trained = set(dm.values()) | tg
    trained -= us
    cop = {m for d, m in dm.items() if D[d]["isCopyrighted"]}
    full = {i for i in trained if i not in cop and i not in tg}
    half = trained - full
    return dm, mm, full, half, us


def save_out(inst, txt, sol, path):
    sc, ok, msg = score_of(txt, sol)
    if ok and strict_ok(inst, sol):
        with open(path, "w") as f:
            json.dump(sol, f)
        return sc
    print("  NOT SAVED", ok, msg, strict_ok(inst, sol))
    return -1


def main():
    path, plan_path = sys.argv[1], sys.argv[2]
    restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rtime = float(sys.argv[5]) if len(sys.argv) > 5 else 60
    outp = sys.argv[6] if len(sys.argv) > 6 else f"solutions/{os.path.basename(path)[:-5]}_v2.json"
    txt = open(path).read()
    inst = json.loads(txt)
    plan = json.load(open(plan_path))
    print("plan obj", plan["obj"], flush=True)
    best = (-1, None)
    t0 = time.time()
    for k in range(restarts):
        rnd = random.Random(k)
        noise = 0.0 if k == 0 else 0.02 * (1 + k % 5)
        order = "lb_asc" if k % 7 != 6 else "lb_desc"
        sol, nf, added = build_solution(inst, plan, rnd, noise, order)
        sc, ok, msg = score_of(txt, sol)
        okk = ok and strict_ok(inst, sol)
        print(f" restart {k} noise {noise:.2f} {order}: failed {nf} added {added} score {sc} {'OK' if okk else msg}"
              f" t={time.time()-t0:.0f}s", flush=True)
        if okk and sc > best[0]:
            best = (sc, sol)
            save_out(inst, txt, sol, outp)
    print("best after restarts", best[0], flush=True)
    # residual rounds
    from select_mip2 import build_and_solve
    sol = best[1]
    for rr in range(rounds):
        dm, mm, full, half, us = classify(inst, sol)
        M, D = inst["models"], inst["datasets"]
        used = full | half | us
        energy = sum(M[i]["energyCost"] for i in used)
        items = {t: ([], []) for t in TYPES}
        for d in D:
            if d["id"] not in dm:
                items[d["dataType"]][d["isCopyrighted"]].append(d["size"])
        rp = build_and_solve(inst, small_L=2500, time_limit=rtime, forced_off=sorted(used),
                             resid_items=items, cap=inst["energyCap"] - energy)
        if rp is None:
            print(" resid infeasible"); break
        print(f" round {rr}: resid obj {rp['obj']:.0f} full {len(rp['full'])} half {len(rp['half'])} src {len(rp['source'])}", flush=True)
        cand = None
        for k in range(6):
            rnd = random.Random(1000 + k)
            s2, nf, added = build_solution(inst, rp, rnd, 0.0 if k == 0 else 0.05, "lb_asc" if k % 2 == 0 else "lb_desc",
                                           exclude=dm.keys(), base=(dict(dm), list(mm), set(full), set(half), set(us)))
            sc, ok, msg = score_of(txt, s2)
            okk = ok and strict_ok(inst, s2)
            print(f"   try {k}: failed {nf} added {added} score {sc} {'OK' if okk else msg}", flush=True)
            if okk and sc > (cand[0] if cand else best[0]):
                cand = (sc, s2)
        if cand is None:
            break
        best = cand
        sol = cand[1]
        save_out(inst, txt, sol, outp)
    print("FINAL", best[0], outp, flush=True)


def polish(inst, sol, max_cand=600, tlimit=240):
    """Add unused models to a finished solution, filling their bins from leftover datasets
    with the swap repair (full mode first, then half)."""
    t0 = time.time()
    M = inst["models"]
    dm, mm, full, half, us = classify(inst, sol)
    energy = sum(M[i]["energyCost"] for i in full | half | us)
    srcreq = [[tg, next(iter(M[s]["dataRequirements"]))] for s, tg in mm]
    used = full | half | us
    cand = [i for i in range(len(M)) if i not in used and M[i]["value"] > 0
            and all(r["upperBound"] >= r["lowerBound"] for r in M[i]["dataRequirements"].values())]
    cand.sort(key=lambda i: -M[i]["value"] / max(M[i]["energyCost"], 1))
    added = 0
    for i in cand[:max_cand]:
        if time.time() - t0 > tlimit:
            break
        if energy + M[i]["energyCost"] > inst["energyCap"]:
            continue
        for mode in ("full", "half"):
            fake = {"full": list(full | ({i} if mode == "full" else set())), "source": [], "srcreq": srcreq}
            left = repair(inst, fake, dm, {i})
            if not left:
                (full if mode == "full" else half).add(i)
                energy += M[i]["energyCost"]
                added += 1
                break
    print(f"  polish added {added} in {time.time()-t0:.0f}s", flush=True)
    return {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": mm}


if __name__ == "__main__":
    main()
