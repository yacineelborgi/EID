"""Pattern set-partitioning MIP for one type.
usage: patmip.py plan.json sol.json t tl out_assign.json [W2] [W1big] [K3] [seed]"""
import sys, json, time, random
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA2'); sys.path.insert(0, '/home/user/EID')
from ptype import type_data, M, D
import highspy
from scipy.sparse import coo_matrix

def build(bins, items, W2=6, W1small=2500, W1big=400, K3=300, K4=200, seed=0, extra=()):
    rng = np.random.default_rng(seed)
    sz = np.array([s for s, _, _ in items]); fr = np.array([f for _, _, f in items], bool)
    idx_all = np.argsort(sz, kind='stable'); idx_fr = idx_all[fr[idx_all]]
    pats = []  # (b, tuple items, waste)
    seen = set()
    def addp(b, js, w):
        key = (b, tuple(sorted(js)))
        if key not in seen:
            seen.add(key); pats.append((b, key[1], w))
    for b, (k, L, U, F) in enumerate(bins):
        idx = idx_fr if F else idx_all
        s = sz[idx]
        hi1 = min(U, L + (W1small if L <= 3000 else W1big))
        a0, a1 = np.searchsorted(s, L), np.searchsorted(s, hi1, side='right')
        for q in range(a0, a1):
            addp(b, (int(idx[q]),), int(s[q] - L))
        npair = 0
        if L >= 2:
            hi2 = min(U, L + W2)
            lo = np.searchsorted(s, L - s); hi = np.searchsorted(s, hi2 - s, side='right')
            for p in range(len(s)):
                st = max(lo[p], p + 1)
                for q in range(st, hi[p]):
                    addp(b, (int(idx[p]), int(idx[q])), int(s[p] + s[q] - L)); npair += 1
        if npair < 60 and L > 6000:
            # sample triples
            n = len(s); got = 0; tries = 0
            while got < K3 and tries < 40:
                tries += 1
                P = rng.integers(0, n, size=(4000, 2))
                ss = s[P[:, 0]] + s[P[:, 1]]
                need = L - ss
                ok = (P[:, 0] != P[:, 1]) & (need > 0) & (need <= s[-1])
                P, need = P[ok], need[ok]
                l0 = np.searchsorted(s, need); l1 = np.searchsorted(s, np.minimum(need + W2, U - (L - need)), side='right')
                for (p, q), x0, x1 in zip(P, l0, l1):
                    if x1 > x0:
                        r = int(rng.integers(x0, x1))
                        if r != p and r != q:
                            js = (int(idx[p]), int(idx[q]), int(idx[r]))
                            addp(b, js, int(sz[list(js)].sum() - L)); got += 1
                            if got >= K3: break
            if L > 25000:
                got = 0; tries = 0
                while got < K4 and tries < 40:
                    tries += 1
                    P = rng.integers(0, n, size=(4000, 3))
                    ss = s[P].sum(1); need = L - ss
                    ok = (P[:, 0] != P[:, 1]) & (P[:, 0] != P[:, 2]) & (P[:, 1] != P[:, 2]) & (need > 0) & (need <= s[-1])
                    P, need = P[ok], need[ok]
                    l0 = np.searchsorted(s, need); l1 = np.searchsorted(s, need + W2, side='right')
                    for pr, x0, x1 in zip(P, l0, l1):
                        if x1 > x0:
                            r = int(rng.integers(x0, x1))
                            if r not in pr:
                                js = tuple(int(idx[x]) for x in pr) + (int(idx[r]),)
                                if sz[list(js)].sum() <= U:
                                    addp(b, js, int(sz[list(js)].sum() - L)); got += 1
                                    if got >= K4: break
    for b, js in extra:
        addp(b, tuple(js), int(sum(items[j][0] for j in js) - bins[b][1]))
    return pats

def solve(bins, items, pats, pen, tl, start=None, gap=0.0, log=True):
    nb, ni, npat = len(bins), len(items), len(pats)
    rows, cols, vals = [], [], []
    for c, (b, js, w) in enumerate(pats):
        rows.append(b); cols.append(c); vals.append(1)
        for j in js:
            rows.append(nb + j); cols.append(c); vals.append(1)
    for b in range(nb):
        rows.append(b); cols.append(npat + b); vals.append(1)
    nv = npat + nb
    A = coo_matrix((vals, (rows, cols)), shape=(nb + ni, nv)).tocsc()
    cost = np.array([w for _, _, w in pats] + list(pen), float)
    h = highspy.Highs()
    h.setOptionValue("output_flag", log); h.setOptionValue("threads", 1); h.setOptionValue("time_limit", tl)
    h.setOptionValue("mip_rel_gap", gap)
    lp = highspy.HighsLp(); lp.num_col_ = nv; lp.num_row_ = nb + ni
    lp.col_cost_ = cost; lp.col_lower_ = np.zeros(nv); lp.col_upper_ = np.ones(nv)
    lp.row_lower_ = np.array([1.0] * nb + [0.0] * ni); lp.row_upper_ = np.ones(nb + ni)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * nv
    h.passModel(lp)
    if start is not None:
        xs = np.zeros(nv)
        key = {(b, js): c for c, (b, js, w) in enumerate(pats)}
        for b in range(nb):
            if b in start: xs[key[(b, tuple(sorted(start[b])))]] = 1
            else: xs[npat + b] = 1
        sol = highspy.HighsSolution(); sol.col_value = list(xs); sol.value_valid = True
        h.setSolution(sol)
    h.run()
    x = np.array(h.getSolution().col_value)
    if len(x) != nv: return None
    asg = {}
    for c in np.nonzero(x[:npat] > 0.5)[0]:
        b, js, w = pats[c]; asg[b] = list(js)
    return asg, h.getInfo().objective_function_value, h.getInfo().mip_dual_bound

if __name__ == "__main__":
    plan = json.load(open(sys.argv[1])); sol = json.load(open(sys.argv[2])); t = sys.argv[3]; tl = float(sys.argv[4]); out = sys.argv[5]
    W2 = int(sys.argv[6]) if len(sys.argv) > 6 else 6
    W1big = int(sys.argv[7]) if len(sys.argv) > 7 else 400
    K3 = int(sys.argv[8]) if len(sys.argv) > 8 else 300
    seed = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    bins, items = type_data(plan, t)
    # start from solution
    dm = {d: m for d, m in sol["dataMappings"]}
    bidx = {b[0]: q for q, b in enumerate(bins)}; jidx = {it[1]: j for j, it in enumerate(items)}
    per = {}
    for d, m in dm.items():
        if D[d]["dataType"] == t and m in bidx: per.setdefault(bidx[m], []).append(jidx[d])
    start = {b: js for b, js in per.items() if bins[b][1] <= sum(items[j][0] for j in js) <= bins[b][2]}
    t0 = time.time()
    pats = build(bins, items, W2=W2, W1big=W1big, K3=K3, seed=seed, extra=list(start.items()))
    print("patterns", len(pats), f"{time.time()-t0:.0f}s", "start waste", sum(sum(items[j][0] for j in js) - bins[b][1] for b, js in start.items()), "start failed", len(bins) - len(start), flush=True)
    pen = [100000.0] * len(bins)
    r = solve(bins, items, pats, pen, tl, start=start)
    asg, obj, bnd = r
    fl = [bins[b] for b in range(len(bins)) if b not in asg]
    print("obj", obj, "bound", bnd, "failed", fl, flush=True)
    json.dump({str(b): js for b, js in asg.items()}, open(out, "w"))
