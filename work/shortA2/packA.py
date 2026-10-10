"""Per-type packer v2: small bins by matching (lam), big bins by randomised exact fill (lb ascending),
end-game MIP over the failed bins + pool (+ donors), then LNS.
usage: packA.py plan.json types lam seeds tmax
"""
import sys, json, time, random
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA2")
from sortedcontainers import SortedList
from ptype import TypePack, mip_assign, match_small
from rfill import seq_fill
from lns import lns_step


def endgame(tp, fail, isF, rng, ndonor=0, tl=20.0):
    """fail: list of bin idx. Re-solve fail + ndonor donors (same class) with pool. Returns True if all filled."""
    bins, items = tp.bins, tp.items
    pool = [j for j in tp.pool() if (items[j][2] or not isF)]
    donors = []
    if ndonor:
        cl = [b for b in tp.asg if bins[b][3] == isF and bins[b][1] > 3000]
        # donors: similar lb to the failed ones, random
        donors = rng.sample(cl, min(ndonor, len(cl)))
    sub = list(fail) + donors
    its = pool + [j for b in donors for j in tp.asg[b]]
    r = mip_assign([bins[b] for b in sub], [items[j] for j in its], tl=tl)
    if r is None:
        return False
    st, out, filled, objv, bnd = r
    for b in donors:
        tp.set_bin(b, [])
    for q, b in enumerate(sub):
        tp.set_bin(b, [its[j] for j in out[q]])
        tp.failed.discard(b)
    return True


def pack_type(plan, t, seed=0, lam=1000.0, theta=3000, w2=2, log=print, tmax=120):
    t0 = time.time()
    tp = TypePack(plan, t, seed, log=lambda s: None)
    rng = tp.rng
    bins, items = tp.bins, tp.items
    small = [b for b in range(len(bins)) if bins[b][1] <= theta]
    r = match_small([bins[b] for b in small], items, lam=lam)
    for q, b in enumerate(small):
        if r[q] is None:
            tp.failed.add(b)
        else:
            tp.set_bin(b, [r[q]])
    for isF in (True, False):
        pool = SortedList((items[j][0], j) for j in tp.pool() if (items[j][2] or not isF))
        bb = [(b, bins[b][1], bins[b][2]) for b in range(len(bins)) if bins[b][1] > theta and bins[b][3] == isF]
        res, fl = seq_fill(bb, pool, rng, 0, w2, "asc")
        for b, its in res.items():
            tp.set_bin(b, [j for _, j in its])
        fail = [x[0] for x in fl]
        tp.failed |= set(fail)
        st = tp.stats()
        log(f"  t{t} {'F' if isF else 'H'}: big {len(bb)} seq-failed {len(fail)} failL {sum(x[1] for x in fl)} pool {sum(s for s, _ in pool)}")
        if fail:
            ok = endgame(tp, fail, isF, rng, 0, tl=20)
            nd = 5
            while not ok and nd <= 40 and time.time() - t0 < tmax:
                ok = endgame(tp, fail, isF, rng, nd, tl=20)
                nd += 5
            log(f"  t{t} {'F' if isF else 'H'}: endgame ok={ok} {tp.stats()} {time.time()-t0:.0f}s")
    # LNS on the rest
    it = 0
    while tp.failed and time.time() - t0 < tmax:
        it += 1
        fl = sorted(tp.failed, key=lambda b: bins[b][1])
        b0s = [rng.choice(fl)]
        nf, dw = lns_step(tp, b0s, ndonor=30, tl=8)
        if nf or dw:
            log(f"   lns {it} filled {nf} freed {dw} -> {tp.stats()}")
    return tp


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    ts = sys.argv[2]
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]
    tmax = float(sys.argv[5]) if len(sys.argv) > 5 else 120
    log = lambda s: print(s, flush=True)
    for t in ts:
        for seed in seeds:
            tp = pack_type(plan, t, seed, lam, log=log, tmax=tmax)
            log(f"t{t} seed {seed} RESULT {tp.stats()} failed {[tp.bins[b] for b in tp.failed]}")
