"""Aggregated model for 7_big with windowed distinct-dataset cuts (sparse form), LP + reduced-cost
fixing + restricted MIP. Columns: f(nM) h(nM) s(nM) U(4).  U_t = sum_{full bins t} lb (normalised by F_t).
Cut (t,j): U_t + sum_{full, lb<s} (s-lb) f <= F_t - P_t(j) + j*s - margin_t,  s = s_{j+1}.
"""
import sys, time, json
import numpy as np
import highspy
from scipy.sparse import coo_matrix
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
from lp7 import Inst, cut_rhs
INF = highspy.kHighsInf

def cut_set(I, centers, half_width=30):
    """one j per distinct s value within +-half_width of s at center j (plus j=0)."""
    cuts = {}
    for t in range(4):
        s = I.fs[t]
        sc = s[centers[t]]
        js = [0]
        for val in range(max(1, int(sc) - half_width), int(sc) + half_width + 1):
            j = int(np.searchsorted(s, val, side="left"))  # first index with s >= val -> s_{j+1}=s[j]
            if j < len(s) and s[j] == val and j > 0:
                js.append(j)
        cuts[t] = sorted(set(js))
    return cuts

class Model:
    def __init__(self, I, cuts, margin=(0, 0, 0, 0), emargin=0.0):
        self.I = I; nM = I.nM; self.cuts = cuts
        v, e = I.v, I.e
        nV = 3 * nM + 4
        c = np.concatenate([-v.astype(float), -(v // 2).astype(float), np.full(nM, 1e-7), np.zeros(4)])
        lo = np.zeros(nV); up = np.ones(nV)
        up[:nM][I.bad | (v <= 0)] = 0
        up[nM:2 * nM][(v // 2) <= 0] = 0
        up[2 * nM:3 * nM][~I.mono | I.bad] = 0
        up[3 * nM:] = 2.0
        self.c, self.lo, self.up, self.nV = c, lo, up, nV
        R, C, V, rlo, rhi = [], [], [], [], []
        idx = np.arange(nM); r = 0
        R += [idx, idx, idx]; C += [idx, idx + nM, idx + 2 * nM]; V += [np.ones(nM)] * 3
        rlo.append(np.full(nM, -INF)); rhi.append(np.ones(nM)); r = nM
        self.r_energy = r
        ef = e / I.cap
        R += [np.full(nM, r)] * 3; C += [idx, idx + nM, idx + 2 * nM]; V += [ef] * 3
        rlo.append([-INF]); rhi.append([1.0 - emargin]); r += 1
        self.r_tot, self.r_src, self.r_U, self.r_cut = {}, {}, {}, {}
        for t in range(4):
            hs = I.has[:, t]; lbt = I.lb[:, t].astype(float); m = hs & ~I.badt[:, t]
            self.r_tot[t] = r
            ii = idx[m]
            R += [np.full(len(ii), r)] * 2; C += [ii, ii + nM]; V += [lbt[ii] / I.T[t]] * 2
            ii = idx[I.mono & ~I.bad & hs]
            R.append(np.full(len(ii), r)); C.append(ii + 2 * nM); V.append(lbt[ii] / I.T[t])
            rlo.append([-INF]); rhi.append([1.0]); r += 1
            self.r_src[t] = r
            ii = idx[I.badt[:, t]]
            R.append(np.full(len(ii), r)); C.append(ii + nM); V.append(np.ones(len(ii)))
            ii = idx[I.mono & ~I.bad & hs]
            R.append(np.full(len(ii), r)); C.append(ii + 2 * nM); V.append(-np.ones(len(ii)))
            rlo.append([-INF]); rhi.append([0.0]); r += 1
            self.r_U[t] = r
            ii = idx[m]
            R.append(np.full(len(ii), r)); C.append(ii); V.append(-lbt[ii] / I.F[t])
            R.append([r]); C.append([3 * nM + t]); V.append([1.0])
            rlo.append([0.0]); rhi.append([0.0]); r += 1
            self.r_cut[t] = []
            for j in cuts[t]:
                Rj, s = cut_rhs(I, t, j)
                sel = m & (I.lb[:, t] < s)
                ii = idx[sel]
                R.append(np.full(len(ii) + 1, r)); C.append(np.concatenate([ii, [3 * nM + t]]))
                V.append(np.concatenate([(s - lbt[ii]) / I.F[t], [1.0]]))
                rlo.append([-INF]); rhi.append([(Rj - margin[t]) / I.F[t]]); self.r_cut[t].append((r, j)); r += 1
        self.nR = r
        self.A = coo_matrix((np.concatenate([np.asarray(x, float) for x in V]),
                             (np.concatenate([np.asarray(x) for x in R]), np.concatenate([np.asarray(x) for x in C]))),
                            shape=(r, nV)).tocsc()
        self.rlo = np.concatenate([np.asarray(x, float) for x in rlo]); self.rhi = np.concatenate([np.asarray(x, float) for x in rhi])

    def solve(self, integer=False, time_limit=600, start=None, gap=1e-9, verbose=False, lo=None, up=None, rhi=None):
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(verbose))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", gap)
        h.setOptionValue("threads", 1)
        lp = highspy.HighsLp(); A = self.A; n = A.shape[1]
        lp.num_col_ = n; lp.num_row_ = A.shape[0]
        lp.col_cost_ = self.c
        lp.col_lower_ = self.lo if lo is None else lo
        lp.col_upper_ = self.up if up is None else up
        lp.row_lower_ = self.rlo; lp.row_upper_ = self.rhi if rhi is None else rhi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
        lp.a_matrix_.num_col_ = n; lp.a_matrix_.num_row_ = A.shape[0]
        if integer:
            integ = [highspy.HighsVarType.kInteger] * (n - 4) + [highspy.HighsVarType.kContinuous] * 4
            lp.integrality_ = integ
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
            out["dual"] = np.array(sol.row_dual)
        return out

    def multipliers(self, dual):
        I = self.I
        y = {"E": max(0.0, -dual[self.r_energy]) / I.cap}
        for t in range(4):
            y[("F", t)] = 0.0
            y[("T", t)] = max(0.0, -dual[self.r_tot[t]]) / I.T[t]
            y[("S", t)] = max(0.0, -dual[self.r_src[t]])
            y[("C", t)] = [(j, max(0.0, -dual[r]) / I.F[t]) for r, j in self.r_cut[t]]
        return y

def profits(I, y):
    nM = I.nM
    g = np.zeros((nM, 4))
    for t in range(4):
        lbt = I.lb[:, t].astype(float)
        u = (y[("F", t)] + y[("T", t)]) * lbt
        for j, c in y[("C", t)]:
            if c > 0: u += c * np.maximum(lbt, float(I.fs[t][j]))
        g[:, t] = np.where(I.has[:, t], u, 0)
    base = y["E"] * I.e.astype(float)
    pf = I.v - base - g.sum(1); pf[I.bad | (I.v <= 0)] = -np.inf
    hu = np.zeros(nM)
    for t in range(4):
        lbt = I.lb[:, t].astype(float)
        cc = np.where(I.badt[:, t], y[("S", t)], np.minimum(y[("T", t)] * lbt, y[("S", t)]))
        hu += np.where(I.has[:, t], cc, 0)
    ph = (I.v // 2) - base - hu; ph[(I.v // 2) <= 0] = -np.inf
    ps = np.full(nM, -np.inf); m = I.mono & ~I.bad; t = I.mtype
    ps[m] = (np.array([y[("S", tt)] for tt in range(4)])[t] - base - np.array([y[("T", tt)] for tt in range(4)])[t] * I.lb[np.arange(nM), t])[m]
    return np.stack([np.zeros(nM), pf, ph, ps], 1)

def plan_from_x(I, x):
    nM = I.nM
    xi = np.round(x[:3 * nM]).astype(int)
    full = np.nonzero(xi[:nM])[0].tolist(); half = np.nonzero(xi[nM:2 * nM])[0].tolist()
    src = np.nonzero(xi[2 * nM:3 * nM])[0].tolist()
    TY = "ntic"
    srcreq = [[i, TY[t]] for i in half for t in range(4) if I.badt[i, t]]
    return {"full": full, "half": half, "source": src, "srcreq": srcreq}

def plan_value(I, p):
    return int(I.v[p["full"]].sum() + (I.v[p["half"]] // 2).sum())

def plan_energy(I, p):
    return int(sum(int(I.e[i]) for i in p["full"] + p["half"] + p["source"]))

def matching_lb(I, p):
    """per type: (sum lb, matching lower bound on consumption of full bins)."""
    out = []
    full = np.array(p["full"], int)
    for t in range(4):
        l = np.sort(I.lb[full[I.has[full, t]], t])
        k = len(l)
        out.append((int(l.sum()), int(np.maximum(l, I.fs[t][:k]).sum())))
    return out
