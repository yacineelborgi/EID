"""Exact-small-bin plan MIP for 2_medium (Strategy B).
f/h/s per model, energy row, free mass rows (big F bins = lb, small F bins = exact item assignment),
tiny-ub H/S bins exact items (any item), source counts per type."""
import json, numpy as np, highspy
from collections import defaultdict
from scipy.sparse import coo_matrix
INST = json.load(open('/home/user/EID/datasets/2_medium.json')); M = INST['models']; D = INST['datasets']
T = 'ntic'
FREE = {t: sorted([(d['size'], d['id']) for d in D if d['dataType'] == t and not d['isCopyrighted']]) for t in T}
ALL = {t: sorted([(d['size'], d['id']) for d in D if d['dataType'] == t]) for t in T}
FT = {t: sum(s for s, _ in FREE[t]) for t in T}
MINF = {t: FREE[t][0][0] for t in T}; MINA = {t: ALL[t][0][0] for t in T}


class Model: pass


def build(theta=3000, W=800, UT=600, marg=None):
    marg = marg or {t: 10 for t in T}
    mo = Model(); nM = len(M)
    cols_lb, cols_ub, cost, integ = [], [], [], []
    def addv(c, ub=1.0):
        cost.append(c); cols_lb.append(0.0); cols_ub.append(ub); return len(cost) - 1
    fv, hv, sv = {}, {}, {}
    unf = {}
    for m in M:
        i = m['id']; R = m['dataRequirements']
        ffill = all(r['upperBound'] >= r['lowerBound'] and r['upperBound'] >= MINF[t] for t, r in R.items())
        U = [t for t, r in R.items() if not (r['upperBound'] >= r['lowerBound'] and r['upperBound'] >= MINA[t])]
        unf[i] = U
        if ffill: fv[i] = addv(m['value'])
        hv[i] = addv(m['value'] // 2)
        if len(R) == 1 and not U: sv[i] = addv(0)
    rows = []  # (dict col->coef, lo, hi, name)
    # exclusivity
    for i in range(nM):
        c = {hv[i]: 1}
        if i in fv: c[fv[i]] = 1
        if i in sv: c[sv[i]] = 1
        if len(c) > 1: rows.append((c, -np.inf, 1, 'x'))
    # energy
    c = {}
    for i in range(nM):
        e = M[i]['energyCost']
        for d in (fv, hv, sv):
            if i in d: c[d[i]] = e
    rows.append((c, -np.inf, INST['energyCap'], 'E'))
    # sources per type
    for t in T:
        c = {}
        for i in range(nM):
            if t in unf[i]: c[hv[i]] = 1
        for i in sv:
            if t in M[i]['dataRequirements']: c[sv[i]] = c.get(sv[i], 0) - 1
        rows.append((c, -np.inf, 0, 'S' + t))
    # mass rows + small bins
    xinfo = []  # (col, binkey(i,t,'F'/'H'), item id, size)
    itemrow = defaultdict(dict)
    mass = {t: {} for t in T}
    sbins = []
    for i in range(nM):
        for t, r in M[i]['dataRequirements'].items():
            L, Ub = r['lowerBound'], r['upperBound']
            if i in fv:
                if L > theta:
                    mass[t][fv[i]] = mass[t].get(fv[i], 0) + L
                else:
                    c = {}
                    for s, did in FREE[t]:
                        if s > min(Ub, L + W): break
                        k = addv(0); xinfo.append((k, (i, t, 'F'), did, s)); c[k] = s
                        mass[t][k] = s; itemrow[did][k] = 1
                    c1 = dict(c); c1[fv[i]] = -L; rows.append((c1, 0, np.inf, 'lo'))
                    c2 = dict(c); c2[fv[i]] = -Ub; rows.append((c2, -np.inf, 0, 'hi'))
                    sbins.append((i, t, 'F'))
            if t not in unf[i] and Ub <= UT:
                own = {hv[i]: 1}
                if i in sv: own[sv[i]] = 1
                c = {}
                for s, did in ALL[t]:
                    if s > Ub: break
                    k = addv(0); xinfo.append((k, (i, t, 'H'), did, s)); c[k] = s; itemrow[did][k] = 1
                    if not D[did]['isCopyrighted']: mass[t][k] = s
                c1 = dict(c); [c1.__setitem__(o, -L) for o in own]; rows.append((c1, 0, np.inf, 'loH'))
                c2 = dict(c); [c2.__setitem__(o, -Ub) for o in own]; rows.append((c2, -np.inf, 0, 'hiH'))
                sbins.append((i, t, 'H'))
    for t in T:
        rows.append((mass[t], -np.inf, FT[t] - marg[t], 'F' + t))
    for did, c in itemrow.items():
        if len(c) > 1: rows.append((c, -np.inf, 1, 'it'))
    r_, c_, v_ = [], [], []
    rlo, rhi = [], []
    for q, (c, lo, hi, nm) in enumerate(rows):
        for k, a in c.items(): r_.append(q); c_.append(k); v_.append(a)
        rlo.append(lo); rhi.append(hi)
    mo.A = coo_matrix((v_, (r_, c_)), shape=(len(rows), len(cost))).tocsc()
    mo.cost = np.array(cost, float); mo.lb = np.array(cols_lb); mo.ub = np.array(cols_ub)
    mo.rlo = np.array(rlo, float); mo.rhi = np.array(rhi, float); mo.rows = rows
    mo.fv, mo.hv, mo.sv, mo.unf, mo.xinfo, mo.sbins = fv, hv, sv, unf, xinfo, sbins
    mo.massrow = {t: [q for q, rr in enumerate(rows) if rr[3] == 'F' + t][0] for t in T}
    return mo


def highs(mo, threads=1):
    h = highspy.Highs(); h.setOptionValue('output_flag', False); h.setOptionValue('threads', threads)
    lp = highspy.HighsLp(); lp.num_col_ = len(mo.cost); lp.num_row_ = mo.A.shape[0]
    lp.col_cost_ = mo.cost; lp.col_lower_ = mo.lb; lp.col_upper_ = mo.ub
    lp.row_lower_ = mo.rlo; lp.row_upper_ = mo.rhi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = mo.A.indptr; lp.a_matrix_.index_ = mo.A.indices; lp.a_matrix_.value_ = mo.A.data
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.integrality_ = [highspy.HighsVarType.kInteger] * len(mo.cost)
    h.passModel(lp)
    return h


def sol_status(S):
    """full/half/source sets from a solution file."""
    tg = {b for _, b in S['modelMappings']}; srcs = {a for a, _ in S['modelMappings']}
    cp = defaultdict(int); used = set()
    for d, m in S['dataMappings']: used.add(m); cp[m] |= D[d]['isCopyrighted']
    used |= tg
    full = {i for i in used - srcs if not cp[i] and i not in tg}
    half = used - srcs - full
    return full, half, srcs


def plan_from_x(mo, x):
    xi = np.round(x).astype(int)
    full = [i for i, k in mo.fv.items() if xi[k]]; half = [i for i, k in mo.hv.items() if xi[k]]
    src = [i for i, k in mo.sv.items() if xi[k]]
    srcreq = [[i, t] for i in half for t in mo.unf[i]]
    small = defaultdict(list)
    for k, b, did, s in mo.xinfo:
        if xi[k]: small[b].append(did)
    val = sum(M[i]['value'] for i in full) + sum(M[i]['value'] // 2 for i in half)
    return {'full': sorted(full), 'half': sorted(half), 'source': sorted(src), 'srcreq': srcreq, 'obj': val,
            'small': {f'{i},{t},{k}': v for (i, t, k), v in small.items()}}
