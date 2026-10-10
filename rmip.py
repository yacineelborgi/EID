"""Selection model (same as select_mip.py) solved with highspy directly: LP duals,
variable fixing by reduced cost and a restricted MIP with an optional warm start."""
import numpy as np
import highspy
from scipy.sparse import coo_matrix

TYPES = "ntic"
INF = highspy.kHighsInf


def build(inst, F, T, cap, spare=None):
    M = inst["models"]
    nM = len(M)
    spare = spare or {t: 0 for t in TYPES}
    bad = np.zeros(nM, bool)
    mono = np.zeros(nM, bool)
    for i, m in enumerate(M):
        reqs = m["dataRequirements"]
        bad[i] = any(r["upperBound"] < r["lowerBound"] for r in reqs.values())
        mono[i] = len(reqs) == 1
    zlist = []
    for i, m in enumerate(M):
        for t, r in m["dataRequirements"].items():
            zlist.append((i, t, r["lowerBound"], r["upperBound"] < r["lowerBound"]))
    nZ = len(zlist)
    nV = 3 * nM + nZ
    fo, ho, so, zo = 0, nM, 2 * nM, 3 * nM
    c = np.zeros(nV)
    for i, m in enumerate(M):
        c[fo + i] = -m["value"]
        c[ho + i] = -(m["value"] // 2)
    c[so:so + nM] = 1e-6
    lbv = np.zeros(nV)
    ubv = np.ones(nV)
    ubv[fo:fo + nM][bad] = 0
    ubv[so:so + nM][(~mono) | bad] = 0
    rows, cols, vals, rlo, rhi = [], [], [], [], []
    r = 0
    for i in range(nM):
        rows += [r, r, r]; cols += [fo + i, ho + i, so + i]; vals += [1, 1, 1]
        rlo.append(-INF); rhi.append(1); r += 1
    for k, (i, t, lb, isb) in enumerate(zlist):
        rows += [r, r]; cols += [zo + k, ho + i]; vals += [1, -1]
        rlo.append(0 if isb else -INF); rhi.append(0); r += 1
    rfree = {t: r + j for j, t in enumerate(TYPES)}; r += 4
    rtot = {t: r + j for j, t in enumerate(TYPES)}; r += 4
    rsrc = {t: r + j for j, t in enumerate(TYPES)}; r += 4
    for t in TYPES:
        rlo.append(-INF); rhi.append(1.0)
    for t in TYPES:
        rlo.append(-INF); rhi.append(1.0)
    for t in TYPES:
        rlo.append(-INF); rhi.append(spare[t])
    for k, (i, t, lb, isb) in enumerate(zlist):
        if not isb:
            rows.append(rfree[t]); cols.append(fo + i); vals.append(lb / F[t])
            rows.append(rtot[t]); cols.append(fo + i); vals.append(lb / T[t])
            rows.append(rtot[t]); cols.append(ho + i); vals.append(lb / T[t])
            rows.append(rtot[t]); cols.append(zo + k); vals.append(-lb / T[t])
        rows.append(rsrc[t]); cols.append(zo + k); vals.append(1)
    for i, m in enumerate(M):
        if mono[i] and not bad[i]:
            t, rr = next(iter(m["dataRequirements"].items()))
            rows.append(rtot[t]); cols.append(so + i); vals.append(rr["lowerBound"] / T[t])
            rows.append(rsrc[t]); cols.append(so + i); vals.append(-1)
    for i, m in enumerate(M):
        e = m["energyCost"] / cap
        for o in (fo, ho, so):
            rows.append(r); cols.append(o + i); vals.append(e)
    rlo.append(-INF); rhi.append(1.0); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nV)).tocsc()
    return dict(c=c, A=A, rlo=np.array(rlo, float), rhi=np.array(rhi, float), lbv=lbv, ubv=ubv,
                zlist=zlist, nM=nM, nV=nV)


def run(mod, lbv=None, ubv=None, integer=False, time_limit=60, start=None, gap=1e-9, verbose=False,
        threads=1):
    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(verbose))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", gap)
    h.setOptionValue("threads", threads)
    lp = highspy.HighsLp()
    A = mod["A"]
    lp.num_col_ = mod["nV"]
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = mod["c"]
    lp.col_lower_ = mod["lbv"] if lbv is None else lbv
    lp.col_upper_ = mod["ubv"] if ubv is None else ubv
    lp.row_lower_ = mod["rlo"]
    lp.row_upper_ = mod["rhi"]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    lp.a_matrix_.num_col_ = mod["nV"]
    lp.a_matrix_.num_row_ = A.shape[0]
    if integer:
        lp.integrality_ = [highspy.HighsVarType.kInteger] * mod["nV"]
    h.passModel(lp)
    if start is not None:
        s = highspy.HighsSolution()
        s.col_value = list(map(float, start))
        s.value_valid = True
        h.setSolution(s)
    h.run()
    sol = h.getSolution()
    info = h.getInfo()
    x = np.array(sol.col_value)
    out = dict(x=x, obj=-info.objective_function_value, status=h.modelStatusToString(h.getModelStatus()))
    if integer:
        out["bound"] = -info.mip_dual_bound
    else:
        out["rc"] = np.array(sol.col_dual)
        out["dual"] = np.array(sol.row_dual)
    return out


def to_plan(mod, x):
    nM = mod["nM"]
    xi = np.round(x).astype(int)
    plan = {"full": [], "half": [], "source": [], "srcreq": []}
    for i in range(nM):
        if xi[i]: plan["full"].append(i)
        if xi[nM + i]: plan["half"].append(i)
        if xi[2 * nM + i]: plan["source"].append(i)
    for k, (i, t, lb, isb) in enumerate(mod["zlist"]):
        if xi[3 * nM + k]:
            plan["srcreq"].append([i, t])
    return plan


def from_solution(inst, mod, sol):
    """Vector x for an existing solution (warm start)."""
    M, D = inst["models"], inst["datasets"]
    nM = mod["nM"]
    x = np.zeros(mod["nV"])
    srcs = {s for s, _ in sol["modelMappings"]}
    tg = {}
    for s, t in sol["modelMappings"]:
        tg.setdefault(t, set()).add(next(iter(M[s]["dataRequirements"])))
    defiled = set()
    used = set()
    for d, m in sol["dataMappings"]:
        used.add(m)
        if D[d]["isCopyrighted"]:
            defiled.add(m)
    used |= set(tg)
    for i in used - srcs:
        if i in tg or i in defiled:
            x[nM + i] = 1
        else:
            x[i] = 1
    for i in srcs:
        x[2 * nM + i] = 1
    for k, (i, t, lb, isb) in enumerate(mod["zlist"]):
        if i in tg and t in tg[i]:
            x[3 * nM + k] = 1
    return x
