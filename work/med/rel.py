"""Rigorous aggregated relaxation R of the Responsible-AI problem (with sources).

Per model i the options are: off, full, half with a sourced subset Z of its requirements
(Z must contain every requirement with ub < lb), source (mono-requirement, ub >= lb).

Coupling rows (all valid for every legitimate solution):
  energy  : sum e_i [i used]                                   <= cap
  free_t  : sum_{full, non-tiny req t} lb + c_t                <= F_t
  cut_tj  : -c_t + s_j * k_t                                   <= -(P_j - j s_j)   (k_t = # tiny full bins)
  cutlb_t : -c_t + sum_{tiny full bins} lb                     <= 0
  tot_t   : sum_{unsourced bins t (full/half/source)} lb       <= T_t
  src_t   : #sourced reqs of type t - #sources of type t       <= 0
c_t in [0, F_t].  P_j = sum of the j smallest free items of type t, s_j the j-th (0-based).
Tiny bins of a type: requirements with lb <= tau_t (any tau gives a valid relaxation).
"""
import json
import sys
from fractions import Fraction as Fr
from itertools import combinations

import highspy
import numpy as np
from scipy.sparse import coo_matrix

TYPES = "ntic"
INF = highspy.kHighsInf


def load(path):
    return json.load(open(path))


class Rel:
    def __init__(self, inst, tau=None, ncut=None):
        M, D = inst["models"], inst["datasets"]
        self.inst = inst
        self.M = M
        self.cap = inst["energyCap"]
        self.nM = nM = len(M)
        self.F = {t: 0 for t in TYPES}
        self.T = {t: 0 for t in TYPES}
        for d in D:
            self.T[d["dataType"]] += d["size"]
            if not d["isCopyrighted"]:
                self.F[d["dataType"]] += d["size"]
        self.free = {t: sorted(d["size"] for d in D if d["dataType"] == t and not d["isCopyrighted"])
                     for t in TYPES}
        self.tau = tau if tau is not None else {t: 0 for t in TYPES}
        self.bad = [any(r["upperBound"] < r["lowerBound"] for r in m["dataRequirements"].values()) for m in M]
        self.mono = [len(m["dataRequirements"]) == 1 for m in M]
        # tiny reqs
        self.tiny = {t: [i for i, m in enumerate(M) if t in m["dataRequirements"]
                         and m["dataRequirements"][t]["lowerBound"] <= self.tau[t]
                         and not self.bad[i]] for t in TYPES}
        self.ncut = {}
        for t in TYPES:
            K = len(self.tiny[t])
            self.ncut[t] = min(K, len(self.free[t])) if ncut is None else min(ncut, K, len(self.free[t]))
        # P_j
        self.P = {t: np.concatenate([[0], np.cumsum(self.free[t])]).astype(np.int64) for t in TYPES}

    def istiny(self, i, t):
        r = self.M[i]["dataRequirements"][t]
        return r["lowerBound"] <= self.tau[t] and not self.bad[i]

    # ---------- LP / MIP model ----------
    def build(self):
        M, nM = self.M, self.nM
        zlist = []
        for i, m in enumerate(M):
            for t, r in m["dataRequirements"].items():
                zlist.append((i, t, r["lowerBound"], r["upperBound"] < r["lowerBound"]))
        self.zlist = zlist
        nZ = len(zlist)
        fo, ho, so, zo, co = 0, nM, 2 * nM, 3 * nM, 3 * nM + nZ
        self.off = (fo, ho, so, zo, co)
        nV = co + 4
        c = np.zeros(nV)
        lbv = np.zeros(nV)
        ubv = np.ones(nV)
        for i, m in enumerate(M):
            c[fo + i] = m["value"]
            c[ho + i] = m["value"] // 2
            if self.bad[i]:
                ubv[fo + i] = 0
            if not self.mono[i] or self.bad[i]:
                ubv[so + i] = 0
        for j, t in enumerate(TYPES):
            ubv[co + j] = self.F[t]
        rows, cols, vals, rlo, rhi = [], [], [], [], []
        r = 0
        for i in range(nM):
            rows += [r, r, r]; cols += [fo + i, ho + i, so + i]; vals += [1, 1, 1]
            rlo.append(-INF); rhi.append(1); r += 1
        for k, (i, t, lb, isb) in enumerate(zlist):
            rows += [r, r]; cols += [zo + k, ho + i]; vals += [1, -1]
            rlo.append(0 if isb else -INF); rhi.append(0); r += 1
        self.nlocal = r
        self.coupling = []  # (name, rowindex)
        rowid = {}

        def newrow(name, lo, hi):
            nonlocal r
            rowid[name] = r
            self.coupling.append(name)
            rlo.append(lo); rhi.append(hi); r += 1
            return rowid[name]

        rE = newrow("E", -INF, self.cap / 1e9)
        for i, m in enumerate(M):
            e = m["energyCost"] / 1e9
            for o in (fo, ho, so):
                rows.append(rE); cols.append(o + i); vals.append(e)
        for j, t in enumerate(TYPES):
            rf = newrow(("F", t), -INF, self.F[t])
            rows.append(rf); cols.append(co + j); vals.append(1.0)
            for k, (i, tt, lb, isb) in enumerate(zlist):
                if tt == t and not isb and not self.istiny(i, t):
                    rows.append(rf); cols.append(fo + i); vals.append(lb)
            rt = newrow(("T", t), -INF, self.T[t])
            for k, (i, tt, lb, isb) in enumerate(zlist):
                if tt == t and not isb:
                    rows.append(rt); cols.append(fo + i); vals.append(lb)
                    rows.append(rt); cols.append(ho + i); vals.append(lb)
                    rows.append(rt); cols.append(zo + k); vals.append(-lb)
            for i in range(nM):
                if self.mono[i] and not self.bad[i] and t in M[i]["dataRequirements"]:
                    rows.append(rt); cols.append(so + i); vals.append(M[i]["dataRequirements"][t]["lowerBound"])
            rs = newrow(("S", t), -INF, 0)
            for k, (i, tt, lb, isb) in enumerate(zlist):
                if tt == t:
                    rows.append(rs); cols.append(zo + k); vals.append(1)
            for i in range(nM):
                if self.mono[i] and not self.bad[i] and t in M[i]["dataRequirements"]:
                    rows.append(rs); cols.append(so + i); vals.append(-1)
            # tiny cuts
            if self.tiny[t]:
                rl = newrow(("L", t), -INF, 0)
                rows.append(rl); cols.append(co + j); vals.append(-1)
                for i in self.tiny[t]:
                    rows.append(rl); cols.append(fo + i); vals.append(M[i]["dataRequirements"][t]["lowerBound"])
                s = self.free[t]
                for jj in range(self.ncut[t]):
                    rc = newrow(("C", t, jj), -INF, -(int(self.P[t][jj]) - jj * s[jj]))
                    rows.append(rc); cols.append(co + j); vals.append(-1)
                    for i in self.tiny[t]:
                        rows.append(rc); cols.append(fo + i); vals.append(s[jj])
        self.rowid = rowid
        A = coo_matrix((vals, (rows, cols)), shape=(r, nV)).tocsc()
        self.A, self.c, self.lbv, self.ubv = A, c, lbv, ubv
        self.rlo, self.rhi = np.array(rlo, float), np.array(rhi, float)
        self.nV = nV

    def solve(self, integer=False, time_limit=60, verbose=False, gap=0.0, lbv=None, ubv=None, start=None,
              objlim=None):
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(verbose))
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("threads", 1)
        h.setOptionValue("mip_rel_gap", gap)
        h.setOptionValue("mip_abs_gap", 0.0)
        if objlim is not None:
            h.setOptionValue("objective_bound", float(objlim))
        lp = highspy.HighsLp()
        A = self.A
        lp.num_col_ = self.nV
        lp.num_row_ = A.shape[0]
        lp.sense_ = highspy.ObjSense.kMaximize
        lp.col_cost_ = self.c
        lp.col_lower_ = self.lbv if lbv is None else lbv
        lp.col_upper_ = self.ubv if ubv is None else ubv
        lp.row_lower_ = self.rlo
        lp.row_upper_ = self.rhi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        lp.a_matrix_.num_col_ = self.nV
        lp.a_matrix_.num_row_ = A.shape[0]
        if integer:
            integ = [highspy.HighsVarType.kInteger] * self.nV
            for j in range(4):
                integ[self.off[4] + j] = highspy.HighsVarType.kContinuous
            lp.integrality_ = integ
        h.passModel(lp)
        if start is not None:
            s = highspy.HighsSolution()
            s.col_value = list(map(float, start))
            s.value_valid = True
            h.setSolution(s)
        h.run()
        sol = h.getSolution()
        info = h.getInfo()
        out = dict(x=np.array(sol.col_value), obj=info.objective_function_value,
                   status=h.modelStatusToString(h.getModelStatus()))
        if integer:
            out["bound"] = info.mip_dual_bound
            out["gap"] = info.mip_gap
        else:
            out["dual"] = np.array(sol.row_dual)
        self.h = h
        return out

    # ---------- exact Lagrangian bound ----------
    def multipliers_from_duals(self, dual):
        """LP row duals (maximization: dual >= 0 for <= rows ... sign as HiGHS) -> u >= 0 dict (Fractions)."""
        u = {}
        for name in self.coupling:
            d = dual[self.rowid[name]]
            v = abs(d) if d * 1 != 0 else 0.0
            # HiGHS: for max problems, duals of binding <= rows have sign; take magnitude, clip tiny
            u[name] = Fr(v) if v > 1e-12 else Fr(0)
        # energy row is scaled by 1e9
        u["E"] = u["E"] / 10 ** 9
        return u

    def options(self, i):
        """List of (opt, value, vec) where vec is dict name -> coefficient (exact ints)."""
        m = self.M[i]
        reqs = m["dataRequirements"]
        v = m["value"]
        e = m["energyCost"]
        res = [("off", 0, {})]
        bad = self.bad[i]
        if not bad:
            vec = {"E": e}
            for t, r in reqs.items():
                lb = r["lowerBound"]
                vec[("T", t)] = vec.get(("T", t), 0) + lb
                if self.istiny(i, t):
                    if ("L", t) in self.rowid:
                        vec[("L", t)] = lb
                        for jj in range(self.ncut[t]):
                            vec[("C", t, jj)] = self.free[t][jj]
                else:
                    vec[("F", t)] = lb
            res.append(("full", v, vec))
        ts = list(reqs)
        badt = [t for t in ts if reqs[t]["upperBound"] < reqs[t]["lowerBound"]]
        for k in range(len(ts) + 1):
            for Z in combinations(ts, k):
                if any(t not in Z for t in badt):
                    continue
                vec = {"E": e}
                for t in ts:
                    if t in Z:
                        vec[("S", t)] = 1
                    else:
                        vec[("T", t)] = reqs[t]["lowerBound"]
                res.append(("half:" + "".join(Z), v // 2, vec))
        if self.mono[i] and not bad:
            t = ts[0]
            res.append(("source", 0, {"E": e, ("T", t): reqs[t]["lowerBound"], ("S", t): -1}))
        return res

    def rhs(self):
        b = {"E": self.cap}
        for t in TYPES:
            b[("F", t)] = self.F[t]
            b[("T", t)] = self.T[t]
            b[("S", t)] = 0
            if ("L", t) in self.rowid:
                b[("L", t)] = 0
                for jj in range(self.ncut[t]):
                    b[("C", t, jj)] = -(int(self.P[t][jj]) - jj * self.free[t][jj])
        return b

    def lagrangian(self, u):
        """Exact upper bound L(u) and per-model reduced profits of every option."""
        b = self.rhs()
        L = sum(u[n] * b[n] for n in u)
        # c_t column: coefficient in objective after dualization = -(u_F - u_L - sum u_C)
        for t in TYPES:
            coef = -u[("F", t)]
            if ("L", t) in u:
                coef += u[("L", t)]
            for jj in range(self.ncut.get(t, 0)):
                if ("C", t, jj) in u:
                    coef += u[("C", t, jj)]
            if coef > 0:
                L += coef * self.F[t]
        red = []
        for i in range(self.nM):
            ops = []
            for name, val, vec in self.options(i):
                p = Fr(val) - sum(u.get(n, 0) * a for n, a in vec.items())
                ops.append((p, name))
            best = max(ops)
            L += best[0]
            red.append(ops)
        return L, red


if __name__ == "__main__":
    inst = load(sys.argv[1])
    tau = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    R = Rel(inst, tau={t: tau for t in TYPES})
    R.build()
    out = R.solve()
    print("LP", out["status"], out["obj"])
    u = R.multipliers_from_duals(out["dual"])
    L, red = R.lagrangian(u)
    print("Lagrangian exact bound", float(L))
