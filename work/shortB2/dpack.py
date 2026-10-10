"""Decomposition packer per type: LSA single-item for small bins (lam guards free mass), exact Packer for big
bins, then end-game LNS. usage: dpack.py plan.json types tmax seed"""
import sys, json, time, random
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA")
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortB2")
from ptype import TypePack, Packer, SortedList
from lns import lns_step


def initial_d(tp, TH=10000, reserve=500, order="desc", K=300, G0=20000, log=print):
    bins, items = tp.bins, tp.items
    nb = len(bins)
    sz = np.array([s for s, _, _ in items], float); fr = np.array([f for _, _, f in items])
    small = [b for b in range(nb) if bins[b][1] <= TH]
    big = [b for b in range(nb) if bins[b][1] > TH]
    bigLF = sum(bins[b][1] for b in big if bins[b][3])
    chosen = None
    for lam in (0.0, 0.1, 0.4, 1.5, 5.0, 20.0, 100.0, 1000.0):
        C = np.full((len(small), len(items)), 1e9)
        for q, b in enumerate(small):
            k, L, U, F = bins[b]
            ok = (sz >= L) & (sz <= U)
            if F:
                ok &= fr
                C[q, ok] = sz[ok] - L
            else:
                C[q, ok] = sz[ok] - L + lam * sz[ok] * fr[ok] * 0.01 + lam * fr[ok]
        r, c = linear_sum_assignment(C)
        used = set(int(j) for q, j in zip(r, c) if C[q, j] < 1e9)
        leftF = sum(sz[j] for j in range(len(items)) if j not in used and fr[j])
        chosen = (lam, r, c, C)
        if leftF >= bigLF + reserve:
            break
    lam, r, c, C = chosen
    for q, j in zip(r, c):
        b = small[q]
        if C[q, j] < 1e9:
            tp.set_bin(b, [int(j)])
        else:
            tp.failed.add(b)
    pool = SortedList((items[j][0], j) for j in tp.pool() if items[j][2])
    pk = Packer([], tp.rng, K=K, G0=G0)
    pk.pool = pool
    key = (lambda b: -bins[b][1]) if order == "desc" else (lambda b: bins[b][1])
    for b in sorted([b for b in big if bins[b][3]], key=key):
        k, L, U, F = bins[b]
        rr = pk.fill(L, U, 0)
        if rr is None:
            rr = pk.fill(L, U, 30)
        if rr is None:
            tp.failed.add(b); continue
        tp.set_bin(b, [j for _, j in rr])
    for j in tp.pool():
        if not items[j][2]:
            pool.add((items[j][0], j))
    for b in sorted([b for b in big if not bins[b][3]], key=key):
        k, L, U, F = bins[b]
        rr = pk.fill(L, U, 0)
        if rr is None:
            rr = pk.fill(L, U, 30)
        if rr is None:
            tp.failed.add(b); continue
        tp.set_bin(b, [j for _, j in rr])
    log(f"  t{tp.t} initial_d lam {lam} {tp.stats()}")
    return tp


def endgame(tp, tmax=120, log=print):
    from fixB import squeeze
    t0 = time.time(); it = 0
    while tp.failed and time.time() - t0 < tmax:
        it += 1
        fl = sorted(tp.failed, key=lambda b: tp.bins[b][1])
        k = min(len(fl), tp.rng.choice([1, 1, 2, 3]))
        b0s = tp.rng.sample(fl, k)
        nd = tp.rng.choice([20, 30, 40])
        nf, dw = lns_step(tp, b0s, ndonor=nd, tl=10)
        if nf or dw:
            log(f"   t{tp.t} it {it} t={time.time()-t0:.0f}s k {k} nd {nd} filled {nf} freed {dw} -> {tp.stats()}")
        if it % 4 == 0:
            fr_ = squeeze(tp, nbins=tp.rng.choice([30, 40, 60]), tl=15)
            if fr_:
                log(f"   t{tp.t} squeeze freed {fr_} -> {tp.stats()}")
    return tp


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1])); ts = sys.argv[2]
    tmax = float(sys.argv[3]); seed = int(sys.argv[4])
    log = lambda s: print(s, flush=True)
    for t in ts:
        for order in ("desc", "asc"):
            tp = TypePack(plan, t, seed, log=lambda s: None)
            initial_d(tp, order=order, log=log)
