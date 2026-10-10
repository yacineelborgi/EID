"""Rigorous aggregated relaxation for 5_energy (energy cap > total energy, so energy is dropped).

Variables: f_i (full), h_i (half), s_i (source, mono only), z_it (req (i,t) of a half model sourced).
For each type t and threshold tau in TAUS (bins with lb <= tau are "tiny"):
  free row : sum_{full, lb>tau} lb f + P_F(k_f) <= F_t           (k_f = # tiny full bins)
  total row: sum_{all bins, lb>tau} lb + G(k_f, k_h) <= F_t + C_t
     G(k_f,k_h) = min_{a>=k_f, a+b=k_f+k_h} P_F(a) + P_C(b)   (tiny bins use distinct items,
     full tiny bins use free items)
P_X(k) = sum of the k smallest items of pool X (convex, tangents at integers).
All coefficients are integers so the safe (Neumaier-Shcherbina) bound can be evaluated exactly.
"""
import json
import sys
from fractions import Fraction

import numpy as np
import highspy
from scipy.sparse import coo_matrix

TYPES = "ntic"
INF = highspy.kHighsInf


class Model:
    def __init__(self):
        self.c, self.lo, self.up, self.integ, self.names = [], [], [], [], []
        self.rows, self.cols, self.vals, self.rlo, self.rhi, self.rnames = [], [], [], [], [], []

    def var(self, c, lo, up, integ, name):
        self.c.append(c); self.lo.append(lo); self.up.append(up); self.integ.append(integ)
        self.names.append(name)
        return len(self.c) - 1

    def row(self, coefs, lo, hi, name=""):
        r = len(self.rlo)
        for j, v in coefs:
            if v != 0:
                self.rows.append(r); self.cols.append(j); self.vals.append(v)
        self.rlo.append(lo); self.rhi.append(hi); self.rnames.append(name)
        return r


def build(inst, taus=(1, 30, 100, 300, 1000, 3000), forbid=None, fix=None):
    """fix: dict var-name -> value (bounds), used for restricted problems."""
    M, D = inst["models"], inst["datasets"]
    nM = len(M)
    assert sum(m["energyCost"] for m in M) <= inst["energyCap"]
    Fs = {t: sorted(d["size"] for d in D if d["dataType"] == t and not d["isCopyrighted"]) for t in TYPES}
    Cs = {t: sorted(d["size"] for d in D if d["dataType"] == t and d["isCopyrighted"]) for t in TYPES}
    F = {t: sum(Fs[t]) for t in TYPES}
    C = {t: sum(Cs[t]) for t in TYPES}
    mdl = Model()
    fv, hv, sv, zv = {}, {}, {}, {}
    for i, m in enumerate(M):
        reqs = m["dataRequirements"]
        bad = any(r["upperBound"] < r["lowerBound"] for r in reqs.values())
        mono = len(reqs) == 1
        v = m["value"]
        if not bad and v > 0:
            fv[i] = mdl.var(v, 0, 1, 1, f"f{i}")
        if v // 2 > 0:
            hv[i] = mdl.var(v // 2, 0, 1, 1, f"h{i}")
        if mono and not bad:
            sv[i] = mdl.var(0, 0, 1, 1, f"s{i}")
        cs = [(fv[i], 1)] if i in fv else []
        cs += [(hv[i], 1)] if i in hv else []
        cs += [(sv[i], 1)] if i in sv else []
        if len(cs) > 1:
            mdl.row(cs, -INF, 1, f"one{i}")
        if i in hv:
            for t, r in reqs.items():
                isb = r["upperBound"] < r["lowerBound"]
                zv[(i, t)] = mdl.var(0, 0, 1, 1, f"z{i}{t}")
                mdl.row([(zv[(i, t)], 1), (hv[i], -1)], 0 if isb else -INF, 0, f"zh{i}{t}")
    # sources per type
    for t in TYPES:
        cs = [(zv[(i, tt)], 1) for (i, tt) in zv if tt == t]
        cs += [(sv[i], -1) for i in sv if next(iter(M[i]["dataRequirements"])) == t]
        mdl.row(cs, -INF, 0, f"src{t}")
    # data rows
    for t in TYPES:
        bins = []  # (lb, kind, var, sign)  kind F = full, H = half/source consumption
        for i, m in enumerate(M):
            r = m["dataRequirements"].get(t)
            if r is None or r["upperBound"] < r["lowerBound"]:
                continue
            lb = r["lowerBound"]
            if i in fv:
                bins.append((lb, "F", fv[i], 1))
            if i in hv:
                bins.append((lb, "H", hv[i], 1))
                bins.append((lb, "H", zv[(i, t)], -1))
            if i in sv:
                bins.append((lb, "H", sv[i], 1))
        for tau in taus:
            big = [b for b in bins if b[0] > tau]
            tiny = [b for b in bins if b[0] <= tau]
            nF = sum(1 for b in tiny if b[1] == "F" and b[3] == 1)
            nT = sum(1 for b in tiny if b[3] == 1)
            # free row
            wF = mdl.var(0, 0, INF, 0, f"wF{t}{tau}")
            mdl.row([(b[2], b[0]) for b in big if b[1] == "F"] + [(wF, 1)], -INF, F[t], f"free{t}{tau}")
            s = Fs[t]
            P = 0
            for j in range(min(nF, len(s))):
                # wF >= P(j) + s[j]*(k - j)
                mdl.row([(wF, 1)] + [(b[2], -s[j]) for b in tiny if b[1] == "F"], P - j * s[j], INF,
                        f"tF{t}{tau}_{j}")
                P += s[j]
            if nF > len(s):
                mdl.row([(b[2], 1) for b in tiny if b[1] == "F"], -INF, len(s), f"cntF{t}{tau}")
            # total row
            if nT == 0:
                mdl.row([(b[2], b[0] * b[3]) for b in big], -INF, F[t] + C[t], f"tot{t}{tau}")
                continue
            a = mdl.var(0, 0, len(Fs[t]), 0, f"a{t}{tau}")
            bb = mdl.var(0, 0, len(Cs[t]), 0, f"b{t}{tau}")
            wA = mdl.var(0, 0, INF, 0, f"wA{t}{tau}")
            wB = mdl.var(0, 0, INF, 0, f"wB{t}{tau}")
            mdl.row([(b[2], b[0] * b[3]) for b in big] + [(wA, 1), (wB, 1)], -INF, F[t] + C[t], f"tot{t}{tau}")
            mdl.row([(a, 1), (bb, 1)] + [(b[2], -b[3]) for b in tiny], 0, 0, f"ab{t}{tau}")
            mdl.row([(a, 1)] + [(b[2], -1) for b in tiny if b[1] == "F"], 0, INF, f"aF{t}{tau}")
            for wv, xv, pool, nmw in ((wA, a, Fs[t], "tA"), (wB, bb, Cs[t], "tB")):
                P = 0
                for j in range(min(nT, len(pool))):
                    mdl.row([(wv, 1), (xv, -pool[j])], P - j * pool[j], INF, f"{nmw}{t}{tau}_{j}")
                    P += pool[j]
    mdl.fv, mdl.hv, mdl.sv, mdl.zv = fv, hv, sv, zv
    return mdl


def to_highs(mdl, integer=True, lo=None, up=None):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    n = len(mdl.c)
    A = coo_matrix((np.array(mdl.vals, float), (mdl.rows, mdl.cols)), shape=(len(mdl.rlo), n)).tocsc()
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = len(mdl.rlo)
    lp.col_cost_ = np.array(mdl.c, float)
    lp.col_lower_ = np.array(mdl.lo if lo is None else lo, float)
    lp.col_upper_ = np.array(mdl.up if up is None else up, float)
    lp.row_lower_ = np.array(mdl.rlo, float)
    lp.row_upper_ = np.array(mdl.rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    lp.sense_ = highspy.ObjSense.kMaximize
    if integer:
        lp.integrality_ = [highspy.HighsVarType.kInteger if x else highspy.HighsVarType.kContinuous
                           for x in mdl.integ]
    h.passModel(lp)
    return h, A


def safe_bound(mdl, y, lo=None, up=None):
    """Exact Lagrangian (Neumaier-Shcherbina) bound for max c x, rlo <= Ax <= rhi, lo <= x <= up,
    for an arbitrary dual vector y (float, converted exactly). Returns (Fraction bound, reduced costs)."""
    lo = mdl.lo if lo is None else lo
    up = mdl.up if up is None else up
    nR = len(mdl.rlo)
    Y = []
    for r in range(nR):
        yr = Fraction(float(y[r]))
        if mdl.rhi[r] == INF and yr > 0:
            yr = Fraction(0)
        if mdl.rlo[r] == -INF and yr < 0:
            yr = Fraction(0)
        Y.append(yr)
    red = [Fraction(int(c)) if float(c).is_integer() else Fraction(c) for c in mdl.c]
    for r, j, v in zip(mdl.rows, mdl.cols, mdl.vals):
        if Y[r]:
            red[j] -= Y[r] * int(v)
    B = Fraction(0)
    for r in range(nR):
        if Y[r] > 0:
            B += Y[r] * int(mdl.rhi[r])
        elif Y[r] < 0:
            B += Y[r] * int(mdl.rlo[r])
    for j in range(len(red)):
        rj = red[j]
        if rj > 0:
            if up[j] == INF:
                return None, red
            B += rj * int(up[j])
        elif rj < 0:
            B += rj * int(lo[j])
    return B, red


if __name__ == "__main__":
    import time
    inst = json.load(open(sys.argv[1]))
    taus = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (1, 30, 100, 300, 1000, 3000)
    t0 = time.time()
    mdl = build(inst, taus)
    print("vars", len(mdl.c), "rows", len(mdl.rlo), f"{time.time() - t0:.1f}s", flush=True)
    h, A = to_highs(mdl, integer=False)
    h.run()
    sol = h.getSolution()
    print("LP obj", h.getInfo().objective_function_value, f"{time.time() - t0:.1f}s", flush=True)
    y = np.array(sol.row_dual)
    # highs: for maximization, row duals sign convention: check both
    B1, _ = safe_bound(mdl, y)
    B2, _ = safe_bound(mdl, -y)
    print("safe bound", float(B1), float(B2) if B2 is not None else None, f"{time.time() - t0:.1f}s")
