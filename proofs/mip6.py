"""Shortage MIP with multi-threshold tiny-bin cuts (valid relaxation).
For each pool P in {F (free, full bins), A (all, all data bins)}, type t and threshold th:
   sum_{bins b, L_b > th} L_b x_b + W  <= Cap_P,t ,   W >= S_P,t(k_th),  k_th = #{bins with L_b <= th}
S = sum of the k smallest items of the pool (convex) -> tangent rows.
"""
import json, sys, time
import numpy as np
import highspy
from scipy.sparse import coo_matrix

TYPES = "ntic"
SC = 1e-4
THETAS = (1, 30, 100, 200, 300, 450, 600, 800, 1000, 1300, 1700, 2200, 2800, 3500, 4500)


class Model:
    pass


def build(inst, thetas=THETAS, cap=None, jmax=450, forced_off=(), extra_bins=None, resid=None, marg=None, exact=False):
    M = inst["models"]; D = inst["datasets"]
    cap = inst["energyCap"] if cap is None else cap
    nM = len(M)
    if resid is None:
        fs = {t: sorted(d["size"] for d in D if d["dataType"] == t and not d["isCopyrighted"]) for t in TYPES}
        al = {t: sorted(d["size"] for d in D if d["dataType"] == t) for t in TYPES}
    else:
        fs = {t: sorted(resid[t][0]) for t in TYPES}
        al = {t: sorted(resid[t][0] + resid[t][1]) for t in TYPES}
    bad = np.zeros(nM, bool); mono = np.zeros(nM, bool)
    for i, m in enumerate(M):
        rq = m["dataRequirements"]
        bad[i] = any(r["upperBound"] < r["lowerBound"] for r in rq.values())
        mono[i] = len(rq) == 1
    off = np.zeros(nM, bool); off[list(forced_off)] = True
    zlist = []
    for i, m in enumerate(M):
        for t, r in m["dataRequirements"].items():
            zlist.append((i, t, r["lowerBound"], r["upperBound"] < r["lowerBound"]))
    nZ = len(zlist)
    fo, ho, so, zo = 0, nM, 2 * nM, 3 * nM
    nV = 3 * nM + nZ
    extra = []  # continuous vars: (name)
    cost = np.zeros(nV)
    for i, m in enumerate(M):
        cost[fo + i] = m["value"]; cost[ho + i] = m["value"] // 2
    lb = np.zeros(nV); ub = np.ones(nV)
    ub[fo:fo + nM][bad | off] = 0
    ub[ho:ho + nM][off] = 0
    ub[so:so + nM][(~mono) | bad | off] = 0
    rows, cols, vals, rlo, rhi = [], [], [], [], []
    r = [0]
    names = []

    def add_row(ent, lo, hi, nm=""):
        for j, v in ent:
            rows.append(r[0]); cols.append(j); vals.append(v)
        rlo.append(lo); rhi.append(hi); names.append(nm); r[0] += 1

    newv = []  # (lb, ub) for extra continuous vars

    def new_var(l=0.0, u=np.inf):
        newv.append((l, u)); return nV + len(newv) - 1

    for i in range(nM):
        add_row([(fo + i, 1), (ho + i, 1), (so + i, 1)], -np.inf, 1, "one")
    for k, (i, t, L, isb) in enumerate(zlist):
        add_row([(zo + k, 1), (ho + i, -1)], 0 if isb else -np.inf, 0, "z")
    # bins
    binsF = {t: [] for t in TYPES}  # (L, [(var, coef)])
    binsA = {t: [] for t in TYPES}
    for k, (i, t, L, isb) in enumerate(zlist):
        if isb:
            continue
        binsF[t].append((L, [(fo + i, 1)]))
        ent = [(fo + i, 1), (ho + i, 1), (zo + k, -1)]
        if mono[i]:
            ent.append((so + i, 1))
        binsA[t].append((L, ent))
    for t in TYPES:
        add_row([(zo + k, 1) for k, (i, tt, L, isb) in enumerate(zlist) if tt == t] +
                [(so + i, -1) for i in range(nM) if mono[i] and not bad[i] and t in M[i]["dataRequirements"]],
                -np.inf, 0, "src_" + t)
    fam = []
    for pool, bins, items in (("F", binsF, fs), ("A", binsA, al)):
        for t in TYPES:
            its = items[t]
            capP = sum(its) - (marg or {}).get((pool, t), 0)
            pre = np.concatenate([[0], np.cumsum(its)])
            # plain
            ent = [(v, L * c * SC) for L, e in bins[t] for v, c in e]
            Pv = new_var(-np.inf, np.inf)
            add_row([(Pv, 1)] + [(v, -c) for v, c in ent], 0, 0, f"P_{pool}_{t}")
            add_row([(Pv, 1)], -np.inf, capP * SC, f"cap_{pool}_{t}")
            fam.append((pool, t, None, Pv, None))
            for th in thetas:
                small = [(L, e) for L, e in bins[t] if L <= th]
                if not small:
                    continue
                K = new_var(0, len(its)); W = new_var(0)
                add_row([(K, 1)] + [(v, -c) for L, e in small for v, c in e], 0, 0, f"K_{pool}_{t}_{th}")
                add_row([(Pv, 1)] + [(v, -L * c * SC) for L, e in small for v, c in e] + [(W, 1)], -np.inf, capP * SC, f"cut_{pool}_{t}_{th}")
                kmax = min(len(its), len(small), jmax)
                for j in range(kmax):
                    sj1 = its[j]
                    add_row([(W, 1), (K, -sj1 * SC)], (pre[j] - j * sj1) * SC, np.inf, f"tan_{pool}_{t}_{th}_{j}")
                fam.append((pool, t, th, K, W))
    # energy
    ent = []
    for i, m in enumerate(M):
        e = m["energyCost"] / cap if not exact else float(m["energyCost"])
        ent += [(fo + i, e), (ho + i, e), (so + i, e)]
    add_row(ent, -np.inf, 1.0 if not exact else float(cap), "energy")
    nTot = nV + len(newv)
    cost = np.concatenate([cost, np.zeros(len(newv))])
    lb = np.concatenate([lb, [a for a, b in newv]])
    ub = np.concatenate([ub, [b for a, b in newv]])
    A = coo_matrix((vals, (rows, cols)), shape=(r[0], nTot)).tocsc()
    mo = Model()
    mo.__dict__.update(dict(A=A, cost=cost, lb=lb, ub=ub, rlo=np.array(rlo, float), rhi=np.array(rhi, float),
                            nV=nV, nTot=nTot, nM=nM, zlist=zlist, fo=fo, ho=ho, so=so, zo=zo, names=names, fam=fam,
                            fs=fs, al=al))
    return mo


def to_highs(mo, integer=True, threads=1, verbose=False):
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    h.setOptionValue("threads", threads)
    lp = highspy.HighsLp()
    lp.num_col_ = mo.nTot
    lp.num_row_ = mo.A.shape[0]
    lp.col_cost_ = mo.cost
    lp.col_lower_ = mo.lb
    lp.col_upper_ = mo.ub
    lp.row_lower_ = mo.rlo
    lp.row_upper_ = mo.rhi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = mo.A.indptr
    lp.a_matrix_.index_ = mo.A.indices
    lp.a_matrix_.value_ = mo.A.data
    lp.sense_ = highspy.ObjSense.kMaximize
    if integer:
        integ = [highspy.HighsVarType.kInteger] * mo.nV + [highspy.HighsVarType.kContinuous] * (mo.nTot - mo.nV)
        lp.integrality_ = integ
    h.passModel(lp)
    return h


def plan_from_x(mo, x):
    xi = np.round(x[:mo.nV]).astype(int)
    nM = mo.nM
    plan = {"full": [], "half": [], "source": [], "srcreq": []}
    for i in range(nM):
        if xi[mo.fo + i]: plan["full"].append(i)
        if xi[mo.ho + i]: plan["half"].append(i)
        if xi[mo.so + i]: plan["source"].append(i)
    for k, (i, t, L, isb) in enumerate(mo.zlist):
        if xi[mo.zo + k]:
            plan["srcreq"].append([i, t])
    return plan


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    t0 = time.time()
    mo = build(inst)
    print("built", mo.A.shape, mo.A.nnz, time.time() - t0, flush=True)
    h = to_highs(mo, integer=False, verbose=True)
    h.run()
    print("LP obj", h.getInfo().objective_function_value, time.time() - t0, flush=True)
