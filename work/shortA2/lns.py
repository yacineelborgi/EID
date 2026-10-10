"""LNS packer per type on top of ptype.TypePack.
usage: lns.py plan.json types time_per_type seed
"""
import sys, json, time, random
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA2")
from ptype import TypePack, mip_assign, match_small, Packer, SortedList


def initial2(tp, theta=3000, lam=0.3, K=300, G0=20000, maxw=30):
    bins, items = tp.bins, tp.items
    small = [b for b in range(len(bins)) if bins[b][1] <= theta]
    big = [b for b in range(len(bins)) if bins[b][1] > theta]
    r = match_small([bins[b] for b in small], items, lam=lam)
    for q, b in enumerate(small):
        if r[q] is None:
            tp.failed.add(b)
        else:
            tp.set_bin(b, [r[q]])
    pool = SortedList((items[j][0], j) for j in tp.pool() if items[j][2])
    pk = Packer([], tp.rng, K=K, G0=G0)
    pk.pool = pool
    for b in sorted([b for b in big if bins[b][3]], key=lambda b: bins[b][1]):
        k, L, U, F = bins[b]
        rr = pk.fill(L, U, 0)
        if rr is None:
            rr = pk.fill(L, U, maxw)
        if rr is None:
            tp.failed.add(b); continue
        tp.set_bin(b, [j for _, j in rr])
    for j in tp.pool():
        if not items[j][2]:
            pool.add((items[j][0], j))
    for b in sorted([b for b in big if not bins[b][3]], key=lambda b: bins[b][1]):
        k, L, U, F = bins[b]
        rr = pk.fill(L, U, 0)
        if rr is None:
            rr = pk.fill(L, U, maxw)
        if rr is None:
            tp.failed.add(b); continue
        tp.set_bin(b, [j for _, j in rr])


def waste_of(tp, b):
    return sum(tp.items[j][0] for j in tp.asg[b]) - tp.bins[b][1]


def lns_step(tp, b0s, ndonor=30, tl=8.0, mode=None):
    bins, items, rng = tp.bins, tp.items, tp.rng
    filled = list(tp.asg.keys())
    L0 = max(bins[b][1] for b in b0s)
    anyF = any(bins[b][3] for b in b0s)
    mode = mode or rng.choice(["waste", "near", "rand", "mix"])
    cand = []
    if mode in ("waste", "mix"):
        wb = sorted(filled, key=lambda b: -waste_of(tp, b) * (0.5 + rng.random()))
        cand += wb[:ndonor // 2]
    if mode in ("near", "mix"):
        cand += sorted(filled, key=lambda b: abs(bins[b][1] - L0) * (0.5 + rng.random()))[:ndonor // 2]
    if anyF and rng.random() < 0.5:
        hf = [b for b in filled if not bins[b][3] and any(items[j][2] for j in tp.asg[b])]
        cand += rng.sample(hf, min(len(hf), ndonor // 3))
    cand += rng.sample(filled, min(len(filled), ndonor))
    cand = [b for b in dict.fromkeys(cand) if b not in b0s][:ndonor]
    sub = list(b0s) + cand
    pl = tp.pool()
    its = pl + [j for b in cand for j in tp.asg[b]]
    pos = {j: q for q, j in enumerate(its)}
    start = {q: [pos[j] for j in tp.asg[b]] for q, b in enumerate(sub) if b in tp.asg}
    nb0 = len(b0s)
    opt = set(range(nb0))
    wopt = {q: 2 * bins[b0s[q]][2] + 100000 for q in range(nb0)}
    cur = sum(items[j][0] for b in cand for j in tp.asg[b])
    r = mip_assign([bins[b] for b in sub], [items[j] for j in its], tl=tl, opt=opt, wopt=wopt, start=start)
    if r is None:
        return 0, 0
    st, out, filled_, objv, bnd = r
    newc = sum(items[j][0] for q in range(len(sub)) for j in out[q]) - sum(
        items[j][0] for q in range(nb0) for j in out[q])
    nfill = sum(1 for q in range(nb0) if q in filled_)
    if nfill == 0 and newc >= cur:
        return 0, 0
    for b in sub:
        tp.set_bin(b, [])
    for q, b in enumerate(sub):
        if q < nb0 and q not in filled_:
            continue
        tp.set_bin(b, [its[j] for j in out[q]])
        tp.failed.discard(b)
    return nfill, cur - newc


def run(tp, tmax=300, log=print, ndonor=30, tl=8.0):
    t0 = time.time()
    it = 0
    while tp.failed and time.time() - t0 < tmax:
        it += 1
        fl = sorted(tp.failed, key=lambda b: tp.bins[b][1])
        k = 1 if tp.rng.random() < 0.7 else min(2, len(fl))
        b0s = tp.rng.sample(fl, k)
        nf, dw = lns_step(tp, b0s, ndonor=ndonor, tl=tl)
        if nf or dw:
            log(f"   it {it} t={time.time()-t0:.0f}s filled {nf} freed {dw} -> {tp.stats()}")
    return tp


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    ts = sys.argv[2]
    tmax = float(sys.argv[3]) if len(sys.argv) > 3 else 300
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    log = lambda s: print(s, flush=True)
    for t in ts:
        tp = TypePack(plan, t, seed, log)
        initial2(tp)
        log(f"t{t} initial {tp.stats()}")
        run(tp, tmax, log)
        log(f"t{t} FINAL {tp.stats()} failed {[tp.bins[b] for b in tp.failed]}")
