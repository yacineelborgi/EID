"""Aggregated LP relaxation for 7_big with tiny-bin (distinct-dataset) cuts, cutting-plane loop,
and an exact (Fraction) Lagrangian bound from the duals.

Valid relaxation:
  full_i : all reqs filled with free data -> value v, uses e and >= lb free data per req
  half_i : value v//2, reqs filled with any data (>= lb total) or sourced (bad reqs must be sourced)
  src_i  : mono, not bad; value 0, uses e and >= lb total data of its type
  cut(t,j): sum_{full bins of t} max(lb, s_{j+1}) <= F_t - P_t(j) + j*s_{j+1}
            (bins use distinct free datasets; P_t(j) = sum of j smallest free datasets of type t)
"""
import sys, time
from fractions import Fraction as Fr
import numpy as np
import highspy
from scipy.sparse import coo_matrix
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
from data7 import load
TYPES = "ntic"
INF = highspy.kHighsInf

class Inst:
    def __init__(self):
        d = load()
        self.__dict__.update(d)
        self.cap = int(d["cap"])
        nM = len(self.v); self.nM = nM
        self.bad = ((self.ub < self.lb) & self.has).any(1)
        self.badt = (self.ub < self.lb) & self.has
        self.mono = self.has.sum(1) == 1
        self.mtype = np.argmax(self.has, 1)
        self.fs = [np.sort(self.ds[(self.dt == t) & (self.dc == 0)]) for t in range(4)]
        self.cs = [np.sort(self.ds[(self.dt == t) & (self.dc == 1)]) for t in range(4)]
        self.F = [int(x.sum()) for x in self.fs]
        self.T = [int(self.fs[t].sum() + self.cs[t].sum()) for t in range(4)]
        self.Pf = [np.concatenate([[0], np.cumsum(x)]) for x in self.fs]  # Pf[t][j] = sum of j smallest

def cut_rhs(I, t, j):
    s = int(I.fs[t][j])  # s_{j+1}
    return I.F[t] - int(I.Pf[t][j]) + j * s, s

class LP:
    def __init__(self, I, cuts=None, fix0=None, fix1=None):
        """cuts: dict t -> sorted list of j (j>=1). Column layout: f(nM) h(nM) s(nM)."""
        self.I = I
        nM = I.nM
        self.cuts = cuts or {t: [] for t in range(4)}
        v, e = I.v, I.e
        c = np.concatenate([-v.astype(float), -(v // 2).astype(float), np.full(nM, 1e-7)])
        lo = np.zeros(3 * nM); up = np.ones(3 * nM)
        up[:nM][I.bad | (v <= 0)] = 0
        up[nM:2 * nM][(v // 2) <= 0] = 0
        up[2 * nM:][~I.mono | I.bad] = 0
        if fix0 is not None: up[fix0] = 0
        if fix1 is not None: lo[fix1] = 1
        self.c, self.lo, self.up = c, lo, up
        rows, cols, vals, rlo, rhi = [], [], [], [], []
        idx = np.arange(nM)
        # per-model
        rows += [idx, idx, idx]; cols += [idx, idx + nM, idx + 2 * nM]; vals += [np.ones(nM)] * 3
        rlo.append(np.full(nM, -INF)); rhi.append(np.ones(nM))
        r = nM
        self.r_energy = r
        ef = e / I.cap
        rows += [np.full(nM, r)] * 3; cols += [idx, idx + nM, idx + 2 * nM]; vals += [ef] * 3
        rlo.append([-INF]); rhi.append([1.0]); r += 1
        self.r_free = {}; self.r_tot = {}; self.r_src = {}; self.r_cut = {}
        for t in range(4):
            hs = I.has[:, t]; lbt = I.lb[:, t].astype(float)
            m = hs & ~I.badt[:, t]
            self.r_free[t] = r
            ii = idx[m]; rows.append(np.full(len(ii), r)); cols.append(ii); vals.append(lbt[ii] / I.F[t])
            rlo.append([-INF]); rhi.append([1.0]); r += 1
            self.r_tot[t] = r
            # total: full + half (good reqs, not sourced) + source
            ii = idx[m]
            rows += [np.full(len(ii), r)] * 2; cols += [ii, ii + nM]; vals += [lbt[ii] / I.T[t]] * 2
            ii = idx[I.mono & ~I.bad & hs]
            rows.append(np.full(len(ii), r)); cols.append(ii + 2 * nM); vals.append(lbt[ii] / I.T[t])
            rlo.append([-INF]); rhi.append([1.0]); r += 1
            self.r_src[t] = r
            ii = idx[I.badt[:, t]]
            rows.append(np.full(len(ii), r)); cols.append(ii + nM); vals.append(np.ones(len(ii)))
            ii = idx[I.mono & ~I.bad & hs]
            rows.append(np.full(len(ii), r)); cols.append(ii + 2 * nM); vals.append(-np.ones(len(ii)))
            rlo.append([-INF]); rhi.append([0.0]); r += 1
        for t in range(4):
            self.r_cut[t] = []
            hs = I.has[:, t] & ~I.badt[:, t]; ii = idx[hs]; lbt = I.lb[ii, t]
            for j in self.cuts[t]:
                R, s = cut_rhs(I, t, j)
                rows.append(np.full(len(ii), r)); cols.append(ii); vals.append(np.maximum(lbt, s) / I.F[t])
                rlo.append([-INF]); rhi.append([R / I.F[t]]); self.r_cut[t].append((r, j)); r += 1
        self.nR = r
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, 3 * nM)).tocsc()
        self.A = A
        self.rlo = np.concatenate([np.asarray(x, float) for x in rlo]); self.rhi = np.concatenate([np.asarray(x, float) for x in rhi])

    def solve(self, integer=False, time_limit=600, start=None, gap=1e-9, verbose=False, energy_rhs=None, lo=None, up=None):
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(verbose))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", gap)
        h.setOptionValue("threads", 1)
        lp = highspy.HighsLp()
        A = self.A
        n = A.shape[1]
        lp.num_col_ = n; lp.num_row_ = A.shape[0]
        lp.col_cost_ = self.c
        lp.col_lower_ = self.lo if lo is None else lo
        lp.col_upper_ = self.up if up is None else up
        rhi = self.rhi.copy()
        if energy_rhs is not None: rhi[self.r_energy] = energy_rhs
        lp.row_lower_ = self.rlo; lp.row_upper_ = rhi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
        lp.a_matrix_.num_col_ = n; lp.a_matrix_.num_row_ = A.shape[0]
        if integer:
            lp.integrality_ = [highspy.HighsVarType.kInteger] * n
        h.passModel(lp)
        if start is not None:
            s = highspy.HighsSolution(); s.col_value = list(map(float, start)); s.value_valid = True
            h.setSolution(s)
        h.run()
        sol = h.getSolution(); info = h.getInfo()
        out = dict(x=np.array(sol.col_value), obj=-info.objective_function_value,
                   status=h.modelStatusToString(h.getModelStatus()))
        if integer:
            out["bound"] = -info.mip_dual_bound
        else:
            out["rc"] = np.array(sol.col_dual); out["dual"] = np.array(sol.row_dual)
        return out

    def multipliers(self, dual):
        """Row duals (minimisation of -value) -> nonnegative multipliers in ORIGINAL units (floats)."""
        I = self.I
        y = {"E": max(0.0, -dual[self.r_energy]) / I.cap}
        for t in range(4):
            y[("F", t)] = max(0.0, -dual[self.r_free[t]]) / I.F[t]
            y[("T", t)] = max(0.0, -dual[self.r_tot[t]]) / I.T[t]
            y[("S", t)] = max(0.0, -dual[self.r_src[t]])
            y[("C", t)] = [(j, max(0.0, -dual[r]) / I.F[t]) for r, j in self.r_cut[t]]
        return y

def separate(I, f, t, top=3, mingap=1e-9):
    """Most violated cut indices j for fractional full vector f (cut normalised by F)."""
    hs = I.has[:, t] & ~I.badt[:, t]
    lb = I.lb[hs, t]; w = f[hs]
    o = np.argsort(lb); lb = lb[o]; w = w[o]
    cw = np.concatenate([[0], np.cumsum(w)]); cwl = np.concatenate([[0], np.cumsum(w * lb)])
    base = cwl[-1]
    s = I.fs[t].astype(float)
    N = len(s)
    j = np.arange(N)
    k = np.searchsorted(lb, s, side="left")  # bins with lb < s
    lhs = base + s * cw[k] - cwl[k]
    rhs = I.F[t] - I.Pf[t][:N] + j * s
    viol = (lhs - rhs) / I.F[t]
    order = np.argsort(-viol)
    return [(int(jj), float(viol[jj])) for jj in order[:top] if viol[jj] > mingap]

def exact_bound(I, y):
    """Exact Lagrangian upper bound (Fraction) for nonnegative multipliers y (any floats)."""
    yE = Fr(y["E"])
    yF = [Fr(y[("F", t)]) for t in range(4)]
    yT = [Fr(y[("T", t)]) for t in range(4)]
    yS = [Fr(y[("S", t)]) for t in range(4)]
    cuts = [[(j, Fr(c)) for j, c in y[("C", t)] if c > 0] for t in range(4)]
    B = yE * I.cap
    for t in range(4):
        B += yF[t] * I.F[t] + yT[t] * I.T[t]
        for j, c in cuts[t]:
            R, s = cut_rhs(I, t, j)
            B += c * R
    # g_t(lb) = sum_c y_c * max(lb, s_c)
    gs = []
    for t in range(4):
        cl = sorted((int(I.fs[t][j]), c) for j, c in cuts[t])
        svals = [s for s, _ in cl]
        pre_y = [Fr(0)]; pre_ys = [Fr(0)]
        for s, c in cl:
            pre_y.append(pre_y[-1] + c); pre_ys.append(pre_ys[-1] + c * s)
        gs.append((np.array(svals, np.int64), pre_y, pre_ys))
    def g(t, lb):
        sv, py, pys = gs[t]
        k = int(np.searchsorted(sv, lb, side="right"))  # cuts with s <= lb
        return py[k] * lb + (pys[-1] - pys[k])
    total = Fr(0)
    best_opt = np.zeros(I.nM, np.int8)
    profit = [None] * I.nM
    for i in range(I.nM):
        vi = int(I.v[i]); ei = int(I.e[i])
        base = yE * ei
        best = Fr(0); bo = 0
        ts = np.nonzero(I.has[i])[0]
        if not I.bad[i] and vi > 0:
            u = base
            for t in ts:
                lb = int(I.lb[i, t]); u += (yF[t] + yT[t]) * lb + g(t, lb)
            p = vi - u
            if p > best: best, bo = p, 1
        if vi // 2 > 0:
            u = base
            for t in ts:
                lb = int(I.lb[i, t])
                u += yS[t] if I.badt[i, t] else min(yT[t] * lb, yS[t])
            p = vi // 2 - u
            if p > best: best, bo = p, 2
        if I.mono[i] and not I.bad[i]:
            t = ts[0]
            p = yS[t] - base - yT[t] * int(I.lb[i, t])
            if p > best: best, bo = p, 3
        total += best; best_opt[i] = bo; profit[i] = best
    return B + total, best_opt
