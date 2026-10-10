"""Exact LNS around a 2_medium solution: release R (full models), candidates P become full with exact free packing;
all other non-fixed models are knapsack half items. usage: lnsx.py tmax seed"""
import sys, json, os, time, random, pickle, collections
import numpy as np, highspy
from scipy.sparse import coo_matrix
sys.path.insert(0, "/home/user/EID")
from state import *
from test_solution import getSolutionScore
from solve import strict_ok
from exact_pack import dp_min_subset
n = len(M)
BAD = {i for i, m in enumerate(M) if any(r['upperBound'] < r['lowerBound'] for r in m['dataRequirements'].values())}
RQ = [{t: (r['lowerBound'], r['upperBound']) for t, r in m['dataRequirements'].items()} for m in M]
C = pickle.load(open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med/lpm_fine.npz.cert.pkl', 'rb'))
RCF = [float(x) for x in C['rcs'][:n]]; RCH = [float(x) for x in C['rcs'][n:2 * n]]
log = lambda s: print(s, flush=True)

def check(sol):
    sc, ok, msg = getSolutionScore(json.dumps(sol), TXT, False)
    return sc if (ok and strict_ok(I, sol)) else -1

def best_known():
    try: return check(json.load(open(OUT)))
    except Exception: return -1

def save_if_better(sol):
    sc = check(sol); b = best_known()
    if sc > b:
        tmp = OUT + ".tmpC%d" % os.getpid(); json.dump(sol, open(tmp, "w")); os.replace(tmp, OUT)
        log(f"  *** wrote {sc} to out_lb (prev {b})")
    return sc

class St:
    def __init__(self, sol):
        self.sol = sol
        S, used, src, tg, F, H = load_sol(sol)
        self.used = {i: list(v) for i, v in used.items()}; self.src = src; self.tg = tg; self.F = F; self.H = H
        u = {d for d, _ in sol['dataMappings']}
        self.leftF = {t: [d['id'] for d in D if d['dataType'] == t and not d['isCopyrighted'] and d['id'] not in u] for t in "ntic"}
        self.leftC = {t: [d['id'] for d in D if d['dataType'] == t and d['isCopyrighted'] and d['id'] not in u] for t in "ntic"}
        self.fixedT = set(tg)  # targets of sources: fixed half
        self.val = value(F, H)

def load_sol(sol):
    used = collections.defaultdict(list)
    for d, m in sol['dataMappings']: used[m].append(d)
    src = {s: t for s, t in sol['modelMappings']}
    tg = collections.defaultdict(list)
    for s, t in sol['modelMappings']: tg[t].append(s)
    trained = (set(used) | set(tg)) - set(src)
    F = set(); H = set()
    for m in trained:
        if m in tg or any(D[d]['isCopyrighted'] for d in used[m]): H.add(m)
        else: F.add(m)
    return sol, used, src, tg, F, H

def solve_nb(st, R, P, tl, verbose=False):
    R = set(R); P = set(P) - st.F - set(st.src) - st.fixedT - BAD
    NB = sorted(R | P)
    fixedF = st.F - R
    Ebase = sum(M[i]['energyCost'] for i in fixedF | set(st.src) | st.fixedT)
    hc = [i for i in range(n) if i not in fixedF and i not in st.src and i not in st.fixedT and i not in BAD and M[i]['value'] // 2 > 0]
    pool = {t: list(st.leftF[t]) for t in "ntic"}
    for i in R:
        for d in st.used[i]: pool[D[d]['dataType']].append(d)
    col = {}; cost = []; colname = []
    def nv(name, c):
        col[name] = len(cost); cost.append(c); colname.append(name); return col[name]
    for i in NB: nv(('f', i), M[i]['value'])
    for i in hc: nv(('h', i), M[i]['value'] // 2)
    bins = [(i, t) for i in NB for t in RQ[i]]
    for (i, t) in bins:
        lb, ub = RQ[i][t]
        for d in pool[t]:
            if D[d]['size'] <= ub: nv(('x', d, i), 0)
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
        if name[0] == 'x':
            byd[name[1]].append(j); byb[(name[2], D[name[1]]['dataType'])].append((j, D[name[1]]['size']))
    for d, js in byd.items():
        if len(js) > 1: addrow([(j, 1) for j in js], -np.inf, 1)
    for (i, t) in bins:
        lb, ub = RQ[i][t]; fj = col[('f', i)]
        ent = byb[(i, t)]
        addrow([(j, s) for j, s in ent] + [(fj, -lb)], 0, np.inf)
        addrow([(j, s) for j, s in ent] + [(fj, -ub)], -np.inf, 0)
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
    for i in R:
        for d in st.used[i]: x0[col[('x', d, i)]] = 1
    base = float(np.dot(cost, x0))
    sol = highspy.HighsSolution(); sol.col_value = list(x0); sol.value_valid = True; h.setSolution(sol)
    h.run()
    info = h.getInfo()
    if info.primal_solution_status < 2: return None, base, None, N
    x = np.array(h.getSolution().col_value)
    obj = float(np.dot(cost, np.round(x)))
    res = {name: 1 for name, j in col.items() if x[j] > 0.5}
    return res, base, (obj, info.mip_dual_bound), N

def apply(st, R, res):
    """build new solution from MIP result"""
    newF = (st.F - set(R)) | {k[1] for k in res if k[0] == 'f'}
    newH = {k[1] for k in res if k[0] == 'h'} | st.fixedT
    dm = []
    for i in st.F - set(R):
        for d in st.used[i]: dm.append([d, i])
    for k in res:
        if k[0] == 'x': dm.append([k[1], k[2]])
    usedD = {d for d, _ in dm}
    # sources keep their data
    for s in st.src:
        for d in st.used[s]: dm.append([d, s]); usedD.add(d)
    # half models: keep old assignment if still half and old data not taken; else pack with cop + leftover free
    leftover = {t: set() for t in "ntic"}
    for d in D:
        if d['id'] not in usedD: leftover[d['dataType']].add(d['id'])
    keep = {}
    for i in newH:
        if i in st.H and all(D[d]['id'] in leftover[D[d]['dataType']] for d in st.used[i]):
            keep[i] = st.used[i]
            for d in st.used[i]: leftover[D[d]['dataType']].discard(d)
    for i in sorted(newH - set(keep), key=lambda i: -max(r[0] for r in RQ[i].values())):
        ds = []
        for t, (lb, ub) in RQ[i].items():
            if any(next(iter(M[s]['dataRequirements'])) == t for s in st.tg.get(i, [])): continue
            items = [(D[d]['size'], d) for d in leftover[t]]
            rr = dp_min_subset(items, lb, ub)
            if rr is None: return None
            for _, d in rr: leftover[t].discard(d); ds.append(d)
        keep[i] = ds
    for i, ds in keep.items():
        for d in ds: dm.append([d, i])
    mm = [[s, t] for s, t in st.sol['modelMappings']]
    return {"dataMappings": dm, "modelMappings": mm}
