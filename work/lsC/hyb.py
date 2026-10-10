"""Hybrid LNS: small bins (lb<=SM) exact single-item matching, other bins aggregated per type; verify by exact packing."""
import sys, json, os, time, random, collections
import numpy as np, highspy
from scipy.sparse import coo_matrix
from lnsx import St, M, D, I, CAP, OUT, RQ, BAD, RCF, RCH, check, save_if_better, log, n
from exact_pack import dp_min_subset, pack_type
SM = 3000; WCAP = 1200; ZMIN = 10**9; PW = 3; NPAIR = 25

def solve_h(st, R, P, tl, margin=None, pen=5, verbose=False, allP_half=True):
    margin = margin or {}
    R = set(R); P = set(P) - st.F - set(st.src) - st.fixedT - BAD
    NB = sorted(R | P); fixedF = st.F - R
    Ebase = sum(M[i]['energyCost'] for i in fixedF | set(st.src) | st.fixedT)
    hc = [i for i in range(n) if i not in fixedF and i not in st.src and i not in st.fixedT and i not in BAD and M[i]['value'] // 2 > 0]
    pool = {t: list(st.leftF[t]) for t in "ntic"}
    for i in R:
        for d in st.used[i]: pool[D[d]['dataType']].append(d)
    ptot = {t: sum(D[d]['size'] for d in pool[t]) for t in "ntic"}
    psorted = {t: sorted(pool[t], key=lambda d: D[d]['size']) for t in "ntic"}
    col = {}; cost = []
    def nv(name, c):
        col[name] = len(cost); cost.append(c); return col[name]
    for i in NB: nv(('f', i), M[i]['value'])
    for i in hc: nv(('h', i), M[i]['value'] // 2)
    small = [(i, t) for i in NB for t in RQ[i] if RQ[i][t][0] <= SM]
    big = [(i, t) for i in NB for t in RQ[i] if RQ[i][t][0] > SM]
    for (i, t) in small:
        lb, ub = RQ[i][t]
        for d in psorted[t]:
            s = D[d]['size']
            if s < lb: continue
            if s > ub or s > lb + WCAP: break
            nv(('y', d, i), 0)
        if i in R:
            cur = [d for d in st.used[i] if D[d]['dataType'] == t]
            if len(cur) == 1 and ('y', cur[0], i) not in col: nv(('y', cur[0], i), 0)
        if lb > ZMIN: nv(('z', i, t), 0)
        pats = set()
        if i in R:
            cur = tuple(sorted(d for d in st.used[i] if D[d]['dataType'] == t))
            if len(cur) > 1: pats.add(cur)
        if lb > 300:
            sz = [(D[d]['size'], d) for d in psorted[t] if D[d]['size'] < lb]
            import bisect
            ss = [a for a, _ in sz]; cand = []
            for q, (s1, d1) in enumerate(sz):
                lo = bisect.bisect_left(ss, lb - s1); hi = bisect.bisect_right(ss, lb + PW - s1)
                for q2 in range(max(lo, q + 1), hi):
                    cand.append((min(s1, ss[q2]), s1 + ss[q2] - lb, d1, sz[q2][1]))
            cand.sort(key=lambda c: (c[1], -c[0]))
            for c in cand[:NPAIR]: pats.add(tuple(sorted((c[2], c[3]))))
        for pt in pats: nv(('p', i, t, pt), 0)
    rows, cols, vals, rlo, rhi = [], [], [], [], []
    r = [0]
    def addrow(ent, lo, hi):
        for j, v in ent: rows.append(r[0]); cols.append(j); vals.append(v)
        rlo.append(lo); rhi.append(hi); r[0] += 1
    hset = set(hc)
    for i in NB:
        if i in hset: addrow([(col[('f', i)], 1), (col[('h', i)], 1)], -np.inf, 1)
    byd = collections.defaultdict(list); byb = collections.defaultdict(list)
    for name, j in col.items():
        if name[0] == 'y':
            byd[name[1]].append(j); byb[(name[2], D[name[1]]['dataType'])].append((j, D[name[1]]['size']))
        elif name[0] == 'p':
            for d in name[3]: byd[d].append(j)
            byb[(name[1], name[2])].append((j, sum(D[d]['size'] for d in name[3])))
    for d, js in byd.items():
        if len(js) > 1: addrow([(j, 1) for j in js], -np.inf, 1)
    for (i, t) in small:
        addrow([(j, 1) for j, s in byb[(i, t)]] + ([(col[('z', i, t)], 1)] if ('z', i, t) in col else []) + [(col[('f', i)], -1)], 0, 0)
    for t in "ntic":
        ent = [(col[('f', i)], RQ[i][t][0]) for (i, tt) in big if tt == t]
        for (i, tt) in small:
            if tt != t: continue
            ent += [(j, s) for j, s in byb[(i, t)]] + ([(col[('z', i, t)], RQ[i][t][0] + pen)] if ('z', i, t) in col else [])
        addrow(ent, -np.inf, ptot[t] - margin.get(t, 0))
    ent = [(col[('f', i)], float(M[i]['energyCost'])) for i in NB] + [(col[('h', i)], float(M[i]['energyCost'])) for i in hc]
    addrow(ent, -np.inf, float(CAP - Ebase))
    N = len(cost)
    A = coo_matrix((vals, (rows, cols)), shape=(r[0], N)).tocsc()
    h = highspy.Highs(); h.setOptionValue('output_flag', verbose); h.setOptionValue('threads', 1)
    h.setOptionValue('time_limit', float(tl)); h.setOptionValue('mip_rel_gap', 0.0)
    lp = highspy.HighsLp(); lp.num_col_ = N; lp.num_row_ = r[0]
    lp.col_cost_ = np.array(cost, float); lp.col_lower_ = np.zeros(N); lp.col_upper_ = np.ones(N)
    lp.row_lower_ = np.array(rlo, float); lp.row_upper_ = np.array(rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.sense_ = highspy.ObjSense.kMaximize; lp.integrality_ = [highspy.HighsVarType.kInteger] * N
    h.passModel(lp)
    x0 = np.zeros(N)
    for i in R: x0[col[('f', i)]] = 1
    for i in st.H:
        if ('h', i) in col: x0[col[('h', i)]] = 1
    for (i, t) in small:
        if i in R:
            ds = [d for d in st.used[i] if D[d]['dataType'] == t]
            if len(ds) == 1 and ('y', ds[0], i) in col: x0[col[('y', ds[0], i)]] = 1
            elif ('p', i, t, tuple(sorted(ds))) in col: x0[col[('p', i, t, tuple(sorted(ds)))]] = 1
            elif ('z', i, t) in col: x0[col[('z', i, t)]] = 1
    base = float(np.dot(cost, x0))
    sol = highspy.HighsSolution(); sol.col_value = list(x0); sol.value_valid = True; h.setSolution(sol)
    h.run()
    info = h.getInfo()
    if info.primal_solution_status < 2: return None, base, None, N
    x = np.array(h.getSolution().col_value)
    obj = float(np.dot(cost, np.round(x)))
    res = {name for name, j in col.items() if x[j] > 0.5}
    return res, base, (obj, info.mip_dual_bound), N

def realize(st, R, res, seed=0, log=log):
    """exact packing: fixed bins keep data; y items for small; DP-pack the rest. returns sol or (None, failed)"""
    R = set(R)
    newF = (st.F - R) | {k[1] for k in res if k[0] == 'f'}
    newH = {k[1] for k in res if k[0] == 'h'} | st.fixedT
    dm = {}
    for i in st.F - R:
        for d in st.used[i]: dm[d] = i
    for s in st.src:
        for d in st.used[s]: dm[d] = s
    ysm = {(k[2], D[k[1]]['dataType']): k[1] for k in res if k[0] == 'y'}
    for (i, t), d in ysm.items(): dm[d] = i
    failed = []
    for t in "ntic":
        pool = [(D[d['id']]['size'], d['id']) for d in D if d['dataType'] == t and not d['isCopyrighted'] and d['id'] not in dm]
        bins = [((i, t), RQ[i][t][0], RQ[i][t][1]) for i in newF - (st.F - R) if t in RQ[i] and (i, t) not in ysm]
        best = None
        for sd, order in enumerate(["lb_desc", "lb_asc", "random", "width", "random"]):
            fill, still, left = pack_type(bins, pool, seed=seed + sd, order=order)
            if best is None or len(still) < len(best[1]): best = (fill, still)
            if not still: break
        fill, still = best
        for key, (lb, ub, its) in fill.items():
            for s, d in its: dm[d] = key[0]
        failed += [(k, t) for k in still]
    if failed: return None, failed
    # half models
    leftover = {t: set() for t in "ntic"}
    for d in D:
        if d['id'] not in dm: leftover[d['dataType']].add(d['id'])
    keep = {}
    for i in newH:
        if i in st.H and all(d in leftover[D[d]['dataType']] for d in st.used[i]):
            keep[i] = st.used[i]
            for d in st.used[i]: leftover[D[d]['dataType']].discard(d)
    for i in sorted(newH - set(keep), key=lambda i: -max(r[0] for r in RQ[i].values())):
        ds = []
        for t, (lb, ub) in RQ[i].items():
            if any(next(iter(M[s]['dataRequirements'])) == t for s in st.tg.get(i, [])): continue
            items = [(D[d]['size'], d) for d in leftover[t] if D[d]['isCopyrighted']]
            rr = dp_min_subset(items, lb, ub)
            if rr is None: return None, [('half', i, t)]
            for _, d in rr: leftover[t].discard(d); ds.append(d)
        keep[i] = ds
    for i, ds in keep.items():
        for d in ds: dm[d] = i
    return {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": [list(x) for x in st.sol['modelMappings']]}, []
