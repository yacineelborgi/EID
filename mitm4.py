"""Meet-in-the-middle search for a near-LP-optimal plan (no sources).

Base = every model at its best option under the LP prices. Cheap models (small gap between
their best and alternative options) may deviate. Enumerate low-cost deviation sets on two
halves, then pair them with a KD-tree box query so that the 5 resources (energy + free data
per type) are respected with tiny slack.

usage: python mitm4.py datasets/4_precise.json margin out_plan.json [Lside] [Ltot] [seed] [maxlist]
"""
import json
import random
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

TYPES = "ntic"


def main():
    path = sys.argv[1]
    margin = int(sys.argv[2])
    outf = sys.argv[3]
    Lside = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    Ltot = float(sys.argv[5]) if len(sys.argv) > 5 else 7.0
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    maxlist = int(sys.argv[7]) if len(sys.argv) > 7 else 3_000_000
    rnd = random.Random(seed)
    inst = json.load(open(path))
    M, D = inst["models"], inst["datasets"]
    cap = inst["energyCap"]
    F = {t: 0 for t in TYPES}
    for d in D:
        if not d["isCopyrighted"]:
            F[d["dataType"]] += d["size"]
    li = json.load(open(f"plans/lp_m{margin}.json"))
    lpv = li["lp"]
    R = np.array([cap] + [F[t] - margin for t in TYPES], dtype=np.int64)

    def vec(i, opt):
        if opt == 0:
            return np.zeros(5, np.int64)
        e = M[i]["energyCost"]
        if opt == 2:
            return np.array([e, 0, 0, 0, 0], np.int64)
        r = M[i]["dataRequirements"]
        return np.array([e] + [r[t]["lowerBound"] if t in r else 0 for t in TYPES], np.int64)

    def val(i, opt):
        return 0 if opt == 0 else (M[i]["value"] if opt == 1 else M[i]["value"] // 2)

    # LP prices (lambda) recovered from reduced profits: use the duals implied by mip4 output
    # recompute duals: p_F = v - lam.a_F  => use least squares on a few models (exact anyway)
    ok = li["ok"]
    rc = li["rc"]
    A, b = [], []
    for k, i in enumerate(ok[:400]):
        pf, ph = rc[k]
        A.append(vec(i, 1).astype(float)); b.append(M[i]["value"] - pf)
        A.append(vec(i, 2).astype(float)); b.append(M[i]["value"] // 2 - ph)
    lam = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    print("lambda", lam, flush=True)
    best_opt = {}
    gaps = []
    for k, i in enumerate(ok):
        pf, ph = rc[k]
        opts = sorted([(pf, 1), (ph, 2), (0.0, 0)], reverse=True)
        best_opt[i] = opts[0][1]
        alts = [(opts[0][0] - p, o) for p, o in opts[1:] if opts[0][0] - p <= Ltot]
        if alts:
            gaps.append((alts[0][0], i, alts))
    base_use = np.zeros(5, np.int64)
    base_val = 0
    for i in ok:
        base_use += vec(i, best_opt[i])
        base_val += val(i, best_opt[i])
    r0 = R - base_use
    print("LP", lpv, "base value", base_val, "r0", r0, "lam.r0", float(lam @ r0), flush=True)
    gaps.sort()
    cheap = gaps
    print("deviatable models", len(cheap), flush=True)

    def enumerate_side(items, L, cap_n):
        # arrays: cost, dv, d (5), and a compact encoding of chosen (list of (i,opt)) via parent pointers
        cost = np.zeros(1); dv = np.zeros(1, np.int64); d = np.zeros((1, 5), np.int64)
        par = [np.full(1, -1, np.int64)]; mv = [np.full(1, -1, np.int64)]
        moves = []
        for gap, i, alts in items:
            for c, o in alts:
                if c > L:
                    continue
                sel = np.nonzero(cost + c <= L)[0]
                if len(sel) == 0:
                    continue
                if len(cost) + len(sel) > cap_n:
                    # keep the cheapest
                    sel = sel[np.argsort(cost[sel])[:max(0, cap_n - len(cost))]]
                    if len(sel) == 0:
                        continue
                mid = len(moves)
                moves.append((i, o))
                bo = best_opt[i]
                dd = vec(i, o) - vec(i, bo)
                cost = np.concatenate([cost, cost[sel] + c])
                dv = np.concatenate([dv, dv[sel] + (val(i, o) - val(i, bo))])
                d = np.concatenate([d, d[sel] + dd])
                par.append(sel.astype(np.int64)); mv.append(np.full(len(sel), mid, np.int64))
        return cost, dv, d, np.concatenate(par), np.concatenate(mv), moves

    # NOTE: an item with two alternatives may get both applied on the same path; forbid it by
    # handling alternatives of one item in a single step
    def enumerate_side2(items, L, cap_n):
        cost = np.zeros(1); dv = np.zeros(1, np.int64); d = np.zeros((1, 5), np.int64)
        par = [np.full(1, -1, np.int64)]; mv = [np.full(1, -1, np.int64)]
        moves = []
        for gap, i, alts in items:
            n0 = len(cost)
            newc, newdv, newd, newp, newm = [], [], [], [], []
            bo = best_opt[i]
            for c, o in alts:
                sel = np.nonzero(cost[:n0] + c <= L)[0]
                if len(sel) == 0:
                    continue
                mid = len(moves)
                moves.append((i, o))
                dd = vec(i, o) - vec(i, bo)
                newc.append(cost[sel] + c); newdv.append(dv[sel] + (val(i, o) - val(i, bo)))
                newd.append(d[sel] + dd); newp.append(sel.astype(np.int64)); newm.append(np.full(len(sel), mid, np.int64))
            if newc:
                nc = np.concatenate(newc)
                if n0 + len(nc) > cap_n:
                    return None
                cost = np.concatenate([cost] + newc)
                dv = np.concatenate([dv] + newdv)
                d = np.concatenate([d] + newd)
                par += newp; mv += newm
        return cost, dv, d, np.concatenate(par), np.concatenate(mv), moves

    def path(k, par, mv, moves):
        out = []
        while k > 0:
            out.append(moves[mv[k]])
            k = par[k]
        return out

    t0 = time.time()
    best = None
    rounds = 0
    while time.time() - t0 < float(sys.argv[8]) if len(sys.argv) > 8 else rounds < 1:
        rounds += 1
        items = list(cheap)
        rnd.shuffle(items)
        A_items = items[0::2]
        B_items = items[1::2]
        def side(its):
            L = Lside
            while True:
                r = enumerate_side2(sorted(its), L, maxlist)
                if r is not None:
                    return r, L
                L *= 0.93
        ra, La = side(A_items)
        rb, Lb = side(B_items)
        L = (La, Lb)
        ca, dva, da, pa, ma, mova = ra
        cb, dvb, db, pb, mb, movb = rb
        print(f"round {rounds}: L {L} |A| {len(ca)} |B| {len(cb)} ({time.time() - t0:.1f}s)", flush=True)
        # scaled coordinates: loss units
        sb = db * lam[None, :]
        tree = cKDTree(sb)
        sa = da * lam[None, :]
        tgt = r0 * lam
        # need sa + sb <= tgt and sum(tgt - sa - sb) + ca + cb <= Ltot
        order = np.argsort(ca)
        found = 0
        CH = 200000
        hits = []
        for c0 in range(0, len(order), CH):
            ks = order[c0:c0 + CH]
            bound = Ltot if best is None else min(Ltot, best[0])
            rem = bound - ca[ks]
            ks = ks[rem > 0]
            rem = rem[rem > 0]
            if len(ks) == 0:
                break
            centers = tgt[None, :] - sa[ks] - rem[:, None] / 2
            res = tree.query_ball_point(centers, rem / 2 + 1e-9, p=np.inf)
            for kk, lst in zip(ks, res):
                if lst:
                    hits.append((int(kk), lst))
        print(f"  candidate pairs from {len(hits)} A-entries ({time.time() - t0:.1f}s)", flush=True)
        for k, idx in hits:
            idx = np.array(idx)
            idx = np.array(idx)
            tot = da[k][None, :] + db[idx]
            feas = np.all(tot <= r0[None, :], axis=1)
            if not feas.any():
                continue
            idx = idx[feas]
            loss = ca[k] + cb[idx] + ((r0[None, :] - da[k][None, :] - db[idx]) * lam[None, :]).sum(axis=1)
            j = int(np.argmin(loss))
            ls = float(loss[j])
            if ls <= Ltot and (best is None or ls < best[0] - 1e-9):
                movesel = path(int(k), pa, ma, mova) + path(int(idx[j]), pb, mb, movb)
                best = (ls, movesel)
                found += 1
                # build the plan
                opt = dict(best_opt)
                for i, o in movesel:
                    opt[i] = o
                full = sorted(i for i in ok if opt[i] == 1)
                half = sorted(i for i in ok if opt[i] == 2)
                use = sum((vec(i, opt[i]) for i in ok), np.zeros(5, np.int64))
                v = sum(val(i, opt[i]) for i in ok)
                slack = R - use
                print(f"  loss {ls:.3f} value {v} slack {slack.tolist()} moves {len(movesel)}", flush=True)
                assert np.all(slack >= 0)
                json.dump({"full": full, "half": half, "source": [], "srcreq": [], "obj": int(v),
                           "margin": margin}, open(outf, "w"))
        print(f"round {rounds} done, best {best[0] if best else None} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
