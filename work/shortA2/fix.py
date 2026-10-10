"""Start from a solution of a plan (old pipeline), rebuild per-type packs, LNS-repair failed bins, re-assemble.
usage: fix.py plan.json sol.json tmax_per_type out.json [seed]
"""
import sys, json, time, random, os
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA2")
sys.path.insert(0, "/home/user/EID")
from ptype import TypePack, M, D, INST, mip_assign
from lns import lns_step
from test_solution import getSolutionScore
from solve import strict_ok

TXT = open("/home/user/EID/datasets/6_shortage.json").read()
OUT = "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_lb/6_shortage_submission.json"
CUR = 668811


def best_known():
    b = CUR
    if os.path.exists(OUT):
        s = json.load(open(OUT))
        sc, ok, _ = getSolutionScore(json.dumps(s), TXT, False)
        if ok and strict_ok(INST, s):
            b = max(b, sc)
    return b


def check(sol):
    sc, ok, msg = getSolutionScore(json.dumps(sol), TXT, False)
    used = {m for _, m in sol["dataMappings"]} | {m for x in sol["modelMappings"] for m in x}
    if sum(M[i]["energyCost"] for i in used) > INST["energyCap"]:
        return -1, "energy"
    return sc if (ok and strict_ok(INST, sol)) else -1, msg


def save_if_better(sol, log=print):
    sc, msg = check(sol)
    b = best_known()
    if sc > b:
        json.dump(sol, open(OUT + ".tmp_a2", "w")); os.replace(OUT + ".tmp_a2", OUT)
        json.dump(sol, open("/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA2/best_%d.json" % sc, "w"))
        log(f"  *** wrote {sc} to out_lb (prev {b})")
    return sc


def packs_from_sol(plan, sol, seed=0):
    dm = {d: m for d, m in sol["dataMappings"]}
    tps = {}
    for t in "ntic":
        tp = TypePack(plan, t, seed, log=lambda s: None)
        jidx = {it[1]: j for j, it in enumerate(tp.items)}
        per = {}
        for d, m in dm.items():
            if D[d]["dataType"] == t and m in tp.bidx:
                per.setdefault(tp.bidx[m], []).append(jidx[d])
        for b in range(len(tp.bins)):
            k, L, U, F = tp.bins[b]
            js = per.get(b, [])
            s = sum(tp.items[j][0] for j in js)
            okF = all(tp.items[j][2] for j in js) if F else True
            if js and L <= s <= U and okF:
                tp.set_bin(b, js)
            else:
                tp.failed.add(b)
        tps[t] = tp
    return tps


def assemble(plan, tps):
    full = set(plan["full"]); half = set(plan["half"]); src = set(plan["source"])
    sr = {tuple(x) for x in plan["srcreq"]}
    filled = {}
    for t, tp in tps.items():
        for b, (k, L, U, F) in enumerate(tp.bins):
            filled.setdefault(k, True)
            if b not in tp.asg:
                filled[k] = False
    ok = lambda i: filled.get(i, True)
    fullok = {i for i in full if ok(i)}
    srcok = {t: sorted([i for i in src if ok(i) and t in M[i]["dataRequirements"]], key=lambda i: M[i]["energyCost"]) for t in "ntic"}
    halfok = sorted([i for i in half if ok(i)], key=lambda i: -M[i]["value"])
    mm = []; halfs = set(); used_src = set()
    for i in halfok:
        need = [t for (j, t) in sr if j == i]
        if all(len(srcok[t]) > 0 for t in need):
            for t in need:
                s = srcok[t].pop(0); mm.append([s, i]); used_src.add(s)
            halfs.add(i)
    trained = fullok | halfs
    dmap = []
    for t, tp in tps.items():
        for b, js in tp.asg.items():
            k = tp.bins[b][0]
            if k in trained or k in used_src:
                for j in js:
                    dmap.append([tp.items[j][1], k])
    E = sum(M[i]["energyCost"] for i in trained | used_src)
    sol = {"dataMappings": dmap, "modelMappings": mm}
    return sol, E


def with_add(plan, tps, seed=0):
    import solve2
    sol, E = assemble(plan, tps)
    dm = {d: m for d, m in sol["dataMappings"]}
    mm = [list(x) for x in sol["modelMappings"]]
    us = {a for a, _ in mm}; tg = {b for _, b in mm}
    trained = (set(dm.values()) - us) | tg
    half = {i for i in trained if i in set(plan["half"])}
    full = trained - half
    added = solve2.greedy_add2(INST, dm, mm, full, half, us, {t: [] for t in "ntic"}, random.Random(seed), 0.0)
    sol2 = {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": mm}
    return sol2, added


def rematch_small(tp, theta=3000, log=print):
    from ptype import match_small
    small = [b for b in tp.asg if tp.bins[b][1] <= theta] + [b for b in tp.failed if tp.bins[b][1] <= theta]
    before = sum(tp.items[j][0] for b in small if b in tp.asg for j in tp.asg[b])
    nf_before = sum(1 for b in small if b in tp.failed)
    old = {b: list(tp.asg.get(b, [])) for b in small}
    for b in small:
        tp.set_bin(b, [])
    pl = tp.pool()
    r = match_small([tp.bins[b] for b in small], [tp.items[j] for j in pl], lam=0.0)
    after = sum(tp.items[pl[r[q]]][0] for q in range(len(small)) if r[q] is not None)
    nf_after = sum(1 for q in range(len(small)) if r[q] is None)
    if nf_after > nf_before or (nf_after == nf_before and after >= before):
        for b in small:
            tp.set_bin(b, old[b])
        log(f"  rematch t{tp.t}: no gain ({before} -> {after})")
        return
    for q, b in enumerate(small):
        if r[q] is None:
            tp.failed.add(b)
        else:
            tp.set_bin(b, [pl[r[q]]]); tp.failed.discard(b)
    log(f"  rematch t{tp.t}: small cons {before} -> {after} failed {nf_before}->{nf_after} {tp.stats()}")


def squeeze(tp, nbins=40, tl=15, log=print):
    """re-solve the bins with most waste (+ pool) minimising consumption."""
    bins, items = tp.bins, tp.items
    filled = [b for b in tp.asg if bins[b][1] > 0]
    wb = sorted(filled, key=lambda b: -(sum(items[j][0] for j in tp.asg[b]) - bins[b][1]) * (0.7 + 0.6 * tp.rng.random()))[:nbins]
    pl = tp.pool()
    its = pl + [j for b in wb for j in tp.asg[b]]
    pos = {j: q for q, j in enumerate(its)}
    start = {q: [pos[j] for j in tp.asg[b]] for q, b in enumerate(wb)}
    cur = sum(items[j][0] for b in wb for j in tp.asg[b])
    r = mip_assign([bins[b] for b in wb], [items[j] for j in its], tl=tl, start=start)
    if r is None:
        return 0
    st, out, filled_, objv, bnd = r
    new = sum(items[its[j]][0] for q in range(len(wb)) for j in out[q])
    if new >= cur:
        return 0
    for b in wb:
        tp.set_bin(b, [])
    for q, b in enumerate(wb):
        tp.set_bin(b, [its[j] for j in out[q]])
    return cur - new


def main():
    plan = json.load(open(sys.argv[1])); sol0 = json.load(open(sys.argv[2]))
    tmax = float(sys.argv[3]); out = sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    log = lambda s: print(s, flush=True)
    tps = packs_from_sol(plan, sol0, seed)
    for t, tp in tps.items():
        log(f"t{t} start {tp.stats()} failed {[tp.bins[b] for b in tp.failed]}")
    sol, E = assemble(plan, tps)
    log(f"assembled start score {check(sol)} E-cap {E-INST['energyCap']}")
    for t, tp in tps.items():
        if t not in os.environ.get("TYPES", "ntic"):
            continue
        if tp.failed:
            rematch_small(tp, log=log)
        t0 = time.time(); it = 0
        while tp.failed and time.time() - t0 < tmax:
            it += 1
            if it % 2 == 1:
                fr = squeeze(tp, nbins=tp.rng.choice([30, 40, 60]), tl=15)
                log(f"  t{t} squeeze freed {fr} -> {tp.stats()}")
            fl = sorted(tp.failed, key=lambda b: tp.bins[b][1])
            b0s = [tp.rng.choice(fl)]
            nd = tp.rng.choice([20, 30, 40, 60])
            nf, dw = lns_step(tp, b0s, ndonor=nd, tl=10)
            if nf or dw:
                log(f"  t{t} it {it} t={time.time()-t0:.0f}s nd {nd} filled {nf} freed {dw} -> {tp.stats()}")
                if nf:
                    sol, E = with_add(plan, tps, seed)
                    sc = save_if_better(sol, log)
                    json.dump(sol, open(out, "w"))
                    log(f"  assembled score {sc} E-cap {E-INST['energyCap']}")
        log(f"t{t} end {tp.stats()} failed {[tp.bins[b] for b in tp.failed]}")
    sol, E = assemble(plan, tps)
    sc = save_if_better(sol, log) if E <= INST["energyCap"] else -1
    json.dump(sol, open(out, "w"))
    log(f"FINAL {sc} E-cap {E-INST['energyCap']}")
    sol2, added = with_add(plan, tps, seed)
    sc2 = save_if_better(sol2, log)
    log(f"FINAL+add {sc2} added {added}")
    if sc2 > sc:
        json.dump(sol2, open(out, "w"))


if __name__ == "__main__":
    main()
