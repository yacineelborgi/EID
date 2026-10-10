"""Add exact small-bin single-item matching (F pool) to a mip6 model (exact=True, SC=1).
Returns (A2, cost2, lb2, ub2, rlo2, rhi2, info)."""
import numpy as np
from scipy.sparse import coo_matrix, hstack, vstack, csr_matrix
def add_match(mo, inst, theta=3000, K=80, active=None, types="ntic", mm=None):
    M = inst["models"]; D = inst["datasets"]
    nT = mo.nTot
    xcols = []  # (bin_idx, item_id, size)
    rows, cols, vals, rlo, rhi = [], [], [], [], []
    nr = mo.A.shape[0]; r = 0
    newA_rows = []  # rows touching old columns: list of (old col, coef)
    def addrow(ent_old, ent_new, lo, hi):
        nonlocal r
        for c, v in ent_old: rows.append(r); cols.append(c); vals.append(v)
        for c, v in ent_new: rows.append(r); cols.append(nT + c); vals.append(v)
        rlo.append(lo); rhi.append(hi); r += 1
    info = {}
    for t in types:
        items = sorted((d["size"], d["id"]) for d in D if d["dataType"] == t and not d["isCopyrighted"])
        cap = sum(s for s, _ in items)
        small, big = [], []
        for k, (i, tt, L, isb) in enumerate(mo.zlist):
            if tt != t or isb: continue
            if active is not None and i not in active: continue
            U = M[i]["dataRequirements"][t]["upperBound"]
            (small if L <= theta else big).append((i, L, U))
        item_rows = {}
        bins = []
        for (i, L, U) in small:
            cand = [q for q, (s, _) in enumerate(items) if L <= s <= U][:K]
            ent = []
            for q in cand:
                c = len(xcols); xcols.append((i, t, items[q][1], items[q][0])); ent.append((c, 1.0))
                item_rows.setdefault(q, []).append(c)
            addrow([(mo.fo + i, -1.0)], ent, 0.0, 0.0)
            bins.append((i, L, U, len(cand)))
        for q, cs in item_rows.items():
            addrow([], [(c, 1.0) for c in cs], -np.inf, 1.0)
        # capacity row
        ent_old = [(mo.fo + i, float(L)) for (i, L, U) in big]
        ent_new = [(c, float(xcols[c][3])) for c in range(len(xcols)) if xcols[c][1] == t]
        addrow(ent_old, ent_new, -np.inf, float(cap - (mm or {}).get(t, 0)))
        info[t] = dict(nsmall=len(small), nbig=len(big), nitems=len(item_rows), cap=cap)
    nx = len(xcols)
    B = coo_matrix((vals, (rows, cols)), shape=(r, nT + nx)).tocsr()
    A1 = hstack([mo.A.tocsr(), csr_matrix((nr, nx))]).tocsr()
    A2 = vstack([A1, B]).tocsc()
    cost2 = np.concatenate([mo.cost, np.zeros(nx)])
    lb2 = np.concatenate([mo.lb, np.zeros(nx)]); ub2 = np.concatenate([mo.ub, np.ones(nx)])
    rlo2 = np.concatenate([mo.rlo, rlo]); rhi2 = np.concatenate([mo.rhi, rhi])
    return A2, cost2, lb2, ub2, rlo2, rhi2, xcols, info
