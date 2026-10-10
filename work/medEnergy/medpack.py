"""Strong packer for 2_medium plans: per type TypePack (shortA) + LNS (squeeze/lns_step) until all bins filled.
usage: medpack.py plans.json idx tmax_per_type out.json [seed] [types_order]"""
import sys, json, time, random, os
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA")
sys.path.insert(0, "/home/user/EID")
import ptype
INSTP = "/home/user/EID/datasets/2_medium.json"
TXT = open(INSTP).read(); INST = json.loads(TXT); M, D = INST["models"], INST["datasets"]
ptype.INST = INST; ptype.M = M; ptype.D = D
from ptype import TypePack, mip_assign, match_small
import lns
from lns import lns_step, initial2
from test_solution import getSolutionScore
from solve import strict_ok, greedy_add
OUT = "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_lb/2_medium_submission.json"
CUR = 272320

def check(sol):
    sc, ok, msg = getSolutionScore(json.dumps(sol), TXT, False)
    return sc if (ok and strict_ok(INST, sol)) else -1

def best_known():
    b = CUR
    if os.path.exists(OUT):
        try: b = max(b, check(json.load(open(OUT))))
        except Exception: pass
    return b

def save_if_better(sol, log=print):
    sc = check(sol); b = best_known()
    if sc > b:
        tmp = OUT + ".tmp%d" % os.getpid(); json.dump(sol, open(tmp, "w")); os.replace(tmp, OUT)
        log(f"  *** wrote {sc} to out_lb (prev {b})")
    return sc

def squeeze(tp, nbins=40, tl=15):
    bins, items = tp.bins, tp.items
    filled = [b for b in tp.asg if bins[b][1] > 0]
    wb = sorted(filled, key=lambda b: -(sum(items[j][0] for j in tp.asg[b]) - bins[b][1]) * (0.7 + 0.6 * tp.rng.random()))[:nbins]
    pl = tp.pool(); its = pl + [j for b in wb for j in tp.asg[b]]
    pos = {j: q for q, j in enumerate(its)}
    start = {q: [pos[j] for j in tp.asg[b]] for q, b in enumerate(wb)}
    cur = sum(items[j][0] for b in wb for j in tp.asg[b])
    r = mip_assign([bins[b] for b in wb], [items[j] for j in its], tl=tl, start=start)
    if r is None: return 0
    st, out, filled_, objv, bnd = r
    new = sum(items[its[j]][0] for q in range(len(wb)) for j in out[q])
    if new >= cur: return 0
    for b in wb: tp.set_bin(b, [])
    for q, b in enumerate(wb): tp.set_bin(b, [its[j] for j in out[q]])
    return cur - new

def assemble(plan, tps):
    full = set(plan["full"]); half = set(plan["half"]); src = set(plan["source"])
    sr = {tuple(x) for x in plan["srcreq"]}
    filled = {}
    for t, tp in tps.items():
        for b, (k, L, U, F) in enumerate(tp.bins):
            filled.setdefault(k, True)
            if b not in tp.asg: filled[k] = False
    ok = lambda i: filled.get(i, True)
    fullok = {i for i in full if ok(i)}
    srcok = {t: sorted([i for i in src if ok(i) and t in M[i]["dataRequirements"]], key=lambda i: M[i]["energyCost"]) for t in "ntic"}
    halfok = sorted([i for i in half if ok(i)], key=lambda i: -M[i]["value"])
    mm = []; halfs = set(); used_src = set()
    for i in halfok:
        need = [t for (j, t) in sr if j == i]
        if all(len(srcok[t]) >= need.count(t) for t in need):
            for t in need:
                s = srcok[t].pop(0); mm.append([s, i]); used_src.add(s)
            halfs.add(i)
    trained = fullok | halfs
    dm = {}
    for t, tp in tps.items():
        for b, js in tp.asg.items():
            k = tp.bins[b][0]
            if k in trained or k in used_src:
                for j in js: dm[tp.items[j][1]] = k
    E = sum(M[i]["energyCost"] for i in trained | used_src)
    return dm, mm, trained, halfs, used_src, E

def finish(plan, tps, log):
    dm, mm, trained, halfs, us, E = assemble(plan, tps)
    sol = {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": mm}
    sc = check(sol) if E <= INST["energyCap"] else -1
    cp = {}
    for d, m in dm.items(): cp[m] = cp.get(m, 0) | D[d]["isCopyrighted"]
    tg = {b for _, b in mm}
    h2 = {i for i in trained if cp.get(i, 0) or i in tg}; f2 = trained - h2
    try:
        added = greedy_add(INST, dm, mm, f2, h2, us)
    except Exception as ex:
        added = None; log(f"  greedy_add error {ex}")
    sol2 = {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": mm}
    sc2 = check(sol2)
    best = sol2 if sc2 >= sc else sol
    s = save_if_better(best, log)
    log(f"  assembled {sc} +add {sc2} (added {added}) E-cap {E-INST['energyCap']}")
    return s, best

def pack_type(plan, t, seed, tmax, log):
    tp = TypePack(plan, t, seed, lambda s: None)
    initial2(tp)
    log(f"t{t} initial {tp.stats()}")
    t0 = time.time(); it = 0
    while tp.failed and time.time() - t0 < tmax:
        it += 1
        if it % 3 == 1:
            fr = squeeze(tp, nbins=tp.rng.choice([30, 40, 60]), tl=12)
            if fr: log(f"  t{t} squeeze freed {fr} -> {tp.stats()}")
        fl = sorted(tp.failed, key=lambda b: tp.bins[b][1])
        b0s = [tp.rng.choice(fl)]
        nd = tp.rng.choice([20, 30, 40, 60])
        nf, dw = lns_step(tp, b0s, ndonor=nd, tl=10)
        if nf or dw:
            log(f"  t{t} it {it} t={time.time()-t0:.0f}s nd {nd} filled {nf} freed {dw} -> {tp.stats()}")
    log(f"t{t} end {tp.stats()} failed {[tp.bins[b] for b in tp.failed]} {time.time()-t0:.0f}s")
    return tp

if __name__ == "__main__":
    plans = json.load(open(sys.argv[1])); plans = [plans] if isinstance(plans, dict) else plans
    plan = plans[int(sys.argv[2])]
    tmax = float(sys.argv[3]); out = sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    order = sys.argv[6] if len(sys.argv) > 6 else "itcn"
    log = lambda s: print(s, flush=True)
    log(f"plan obj {plan.get('obj')}")
    tps = {}
    for t in order:
        tps[t] = pack_type(plan, t, seed, tmax, log)
        if tps[t].failed and os.environ.get("STOPFAIL"):
            log("type failed, stop"); sys.exit(0)
    s, best = finish(plan, tps, log)
    json.dump(best, open(out, "w"))
    log(f"FINAL {s}")
