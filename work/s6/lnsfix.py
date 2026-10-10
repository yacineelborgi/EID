"""End-game rescue of failed models: for each failed model and each of its data bins, pool leftover
items with the items of a few donor bins of the same type and solve a small assignment MIP (donors must
stay inside their windows, the rescued bin must be filled). Commit only if all bins of the model succeed."""
import random, sys, time
import numpy as np
import highspy
from scipy.sparse import coo_matrix

sys.path.insert(0, '/home/user/EID')
from assign import TYPES


def solve_assign(bins, items, allowed, tl=10.0):
    """bins: list of (L, U); items: list of sizes; allowed(b, j) -> bool. Returns list of item-index lists or None."""
    nb, ni = len(bins), len(items)
    var = []
    for b in range(nb):
        L, U = bins[b]
        for j in range(ni):
            if items[j] <= U and allowed(b, j):
                var.append((b, j))
    if not var:
        return None
    nv = len(var)
    rows, cols, vals = [], [], []
    for k, (b, j) in enumerate(var):
        rows.append(b); cols.append(k); vals.append(items[j])
        rows.append(nb + j); cols.append(k); vals.append(1)
    A = coo_matrix((vals, (rows, cols)), shape=(nb + ni, nv)).tocsc()
    rlo = np.array([bins[b][0] for b in range(nb)] + [0] * ni, float)
    rhi = np.array([bins[b][1] for b in range(nb)] + [1] * ni, float)
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('threads', 1)
    h.setOptionValue('time_limit', tl)
    lp = highspy.HighsLp()
    lp.num_col_ = nv; lp.num_row_ = nb + ni
    # small objective: minimise total used size (keeps waste low)
    lp.col_cost_ = np.array([items[j] for b, j in var], float)
    lp.col_lower_ = np.zeros(nv); lp.col_upper_ = np.ones(nv)
    lp.row_lower_ = rlo; lp.row_upper_ = rhi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * nv
    h.passModel(lp)
    h.setOptionValue('mip_rel_gap', 1e-5)
    h.run()
    st = h.getModelStatus()
    x = h.getSolution().col_value
    if st not in (highspy.HighsModelStatus.kOptimal,) and not (len(x) == nv and h.getInfo().primal_solution_status == 2):
        return None
    if len(x) != nv:
        return None
    out = [[] for _ in range(nb)]
    for k, (b, j) in enumerate(var):
        if x[k] > 0.5:
            out[b].append(j)
    for b in range(nb):
        s = sum(items[j] for j in out[b])
        if not (bins[b][0] <= s <= bins[b][1]):
            return None
    used = [j for o in out for j in o]
    if len(used) != len(set(used)):
        return None
    return out


import os
def rescue(inst, plan, data_map, failed, rnd, n_donor=int(os.environ.get("NDONOR", 25)), tl=3.0, tbudget=120.0, log=None):
    M, D = inst['models'], inst['datasets']
    t0 = time.time()
    full = set(plan['full'])
    srcreq = {(i, t) for i, t in plan['srcreq']}
    for d in [d for d, m in data_map.items() if m in failed]:
        del data_map[d]
    content = {}
    for d, m in data_map.items():
        content.setdefault((m, D[d]['dataType']), []).append(d)
    rescued = set()
    order = sorted(failed, key=lambda i: -M[i]['value'])
    for i in order:
        if time.time() - t0 > tbudget:
            break
        reqs = [(t, r) for t, r in M[i]['dataRequirements'].items() if (i, t) not in srcreq]
        if any(r['upperBound'] < r['lowerBound'] for _, r in reqs):
            continue
        changes = []  # (type, new assignment dict d->m for touched datasets, old dict)
        ok = True
        for t, r in reqs:
            fr = i in full
            left = [d['id'] for d in D if d['dataType'] == t and d['id'] not in data_map]
            # donors: filled bins of type t, compatible class
            cand = [(m, tt) for (m, tt) in content if tt == t and m != i]
            if fr:
                cand_f = [c for c in cand if c[0] in full]
                cand = cand_f
            else:
                cand = [c for c in cand if c[0] not in full] + [c for c in cand if c[0] in full][:0]
            rnd.shuffle(cand)
            donors = cand[:n_donor]
            items = left + [d for c in donors for d in content[c]]
            sizes = [D[d]['size'] for d in items]
            isfree = [not D[d]['isCopyrighted'] for d in items]
            bins = [(r['lowerBound'], r['upperBound'])] + [(M[m]['dataRequirements'][t]['lowerBound'], M[m]['dataRequirements'][t]['upperBound']) for m, _ in donors]
            bfree = [fr] + [m in full for m, _ in donors]
            res = solve_assign(bins, sizes, lambda b, j: (not bfree[b]) or isfree[j], tl=tl)
            if res is None:
                ok = False
                break
            old = {}
            for c in donors:
                for d in content[c]:
                    old[d] = data_map.get(d)
            for d in left:
                old[d] = None
            # apply
            for c in donors:
                for d in content[c]:
                    del data_map[d]
            newc = {}
            for b, js in enumerate(res):
                m = i if b == 0 else donors[b - 1][0]
                newc[(m, t)] = [items[j] for j in js]
                for j in js:
                    data_map[items[j]] = m
            oldc = {c: content.get(c) for c in newc}
            content.update(newc)
            changes.append((old, oldc))
        if ok:
            rescued.add(i)
            if log: log(f'   rescued {i} value {M[i]["value"]} t={time.time()-t0:.0f}s')
        else:
            for old, oldc in reversed(changes):
                for d, m in old.items():
                    if m is None:
                        data_map.pop(d, None)
                    else:
                        data_map[d] = m
                for c, v in oldc.items():
                    if v is None:
                        content.pop(c, None)
                    else:
                        content[c] = v
    return failed - rescued
