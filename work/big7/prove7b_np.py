"""Reduced-cost-fixed exact-integer MIP over the aggregated relaxation (7_big).

Rigorous part: with LP multipliers y (lp_state.pkl) the exact Lagrangian bound L (Fractions) satisfies
  value(x) = L - sum_i dev_i(x_i) - sum_r y_r slack_r(x)   (dev, slack >= 0 for any feasible x),
so a solution of value >= T uses only options with dev <= L - T.  All other options are removed.
The remaining relaxation (energy, free-data rows, all distinct-dataset cuts of lp_state, source rows;
total-data rows dropped = further relaxation) is solved by HiGHS with integer data and the fixed part
moved to the right-hand side with exact Python integers.  max < T  =>  OPT <= T-1.
usage: prove7.py T [time_limit] [out_plan.json]
"""
import sys, json, pickle, time
from fractions import Fraction as Fr
import numpy as np
import highspy
from scipy.sparse import coo_matrix
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
from lp7 import Inst, exact_bound, cut_rhs
from mip7 import profits
INF = highspy.kHighsInf

T = int(sys.argv[1]); tl = float(sys.argv[2]) if len(sys.argv) > 2 else 600
outp = sys.argv[3] if len(sys.argv) > 3 else None
I = Inst(); nM = I.nM
cuts, y, obj = pickle.load(open("/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7/lp_state.pkl", "rb"))
L, bo = exact_bound(I, y)
print("exact Lagrangian bound", float(L), flush=True)
thr = float(L - T) + 1e-6
P = profits(I, y)
P = np.concatenate([np.zeros((nM, 1)), P[:, 1:]], 1)
best = P.max(1)
dev = best[:, None] - P
allow = (dev <= thr) & np.isfinite(dev)
nopt = allow.sum(1)
free = np.nonzero(nopt > 1)[0]
fixo = np.where(nopt == 1, allow.argmax(1), -1)
assert (nopt >= 1).all()
print("threshold", thr, "free models", len(free), "free options", int(allow[free].sum()), flush=True)
fixed = np.nonzero(nopt == 1)[0]
val = {0: lambda i: 0, 1: lambda i: int(I.v[i]), 2: lambda i: int(I.v[i]) // 2, 3: lambda i: 0}
V0 = sum(val[int(fixo[i])](i) for i in fixed)
E0 = sum(int(I.e[i]) for i in fixed if fixo[i] > 0)
ffull = [int(i) for i in fixed if fixo[i] == 1]
fhalf = [int(i) for i in fixed if fixo[i] == 2]
fsrc = [int(i) for i in fixed if fixo[i] == 3]
cols = [(int(i), o) for i in free for o in (1, 2, 3) if allow[i, o]]
n = len(cols)
R, C, Vv, rlo, rhi = [], [], [], [], []
r = 0
byi = {}
for k, (i, o) in enumerate(cols):
    byi.setdefault(i, []).append(k)
for i, ks in byi.items():
    for k in ks:
        R.append(r); C.append(k); Vv.append(1.0)
    must = not allow[i, 0]
    rlo.append(1.0 if must else -INF); rhi.append(1.0); r += 1
rows_rhs = []
# energy
for k, (i, o) in enumerate(cols):
    R.append(r); C.append(k); Vv.append(float(I.e[i]))
rlo.append(-INF); rhi.append(float(I.cap - E0)); rows_rhs.append(("E", I.cap - E0)); r += 1
for t in range(4):
    lbt = I.lb[:, t]
    # free data row
    fx = sum(int(lbt[i]) for i in ffull if I.has[i, t])
    for k, (i, o) in enumerate(cols):
        if o == 1 and I.has[i, t]:
            R.append(r); C.append(k); Vv.append(float(lbt[i]))
    rlo.append(-INF); rhi.append(float(I.F[t] - fx)); r += 1
    # cuts
    for j in cuts[t]:
        Rj, s = cut_rhs(I, t, j)
        fx = sum(max(int(lbt[i]), s) for i in ffull if I.has[i, t])
        for k, (i, o) in enumerate(cols):
            if o == 1 and I.has[i, t]:
                R.append(r); C.append(k); Vv.append(float(max(int(lbt[i]), s)))
        rlo.append(-INF); rhi.append(float(Rj - fx)); r += 1
    # sources
    need = sum(1 for i in fhalf if I.badt[i, t]); have = sum(1 for i in fsrc if I.mtype[i] == t)
    for k, (i, o) in enumerate(cols):
        if o == 2 and I.badt[i, t]:
            R.append(r); C.append(k); Vv.append(1.0)
        if o == 3 and I.mtype[i] == t:
            R.append(r); C.append(k); Vv.append(-1.0)
    rlo.append(-INF); rhi.append(float(have - need)); r += 1

# ---- implied rows (valid for every solution of value >= T): each Lagrangian term <= budget ----
budget = float(L - T) + 1e-6
# dev knapsack: sum_i dev_i(x_i) <= budget ; for models that may be off: dev_off + sum_o (dev_o - dev_off) x_o
const = 0.0
for i, ks in byi.items():
    d0 = float(dev[i, 0]) if allow[i, 0] else 0.0
    const += d0
    for k in ks:
        o = cols[k][1]
        R.append(r); C.append(k); Vv.append(float(dev[i, o]) - d0)
rlo.append(-INF); rhi.append(budget - const); r += 1
yE = float(y["E"])
if yE > 0:
    w = budget / yE * (1 + 1e-9) + 1
    rlo[rows_rhs and 0 or 0]  # no-op
    # energy row index = first row after the per-model rows
    eidx = len(byi)
    rlo[eidx] = rhi[eidx] - w
cidx = eidx + 1
for t in range(4):
    cidx += 1  # skip free-data row
    for j in cuts[t]:
        c = dict(y[("C", t)]).get(j, 0.0)
        if c > 0:
            rlo[cidx] = rhi[cidx] - (budget / c * (1 + 1e-9) + 1)
        cidx += 1
    if y[("S", t)] > budget:
        rlo[cidx] = rhi[cidx]  # source row must be tight
    cidx += 1
assert cidx == r - 1, (cidx, r)
print("rows", r, "cols", n, "fixed value", V0, "energy rhs", I.cap - E0, flush=True)

import os
if os.environ.get("START"):
    pl = json.load(open(os.environ["START"]))
    o_ = np.zeros(nM, int); o_[pl["full"]] = 1; o_[pl["half"]] = 2; o_[pl["source"]] = 3
    okfix = all(o_[i] == fixo[i] for i in fixed)
    print("start plan: fixed models agree", okfix, "n disagree", int(sum(o_[i] != fixo[i] for i in fixed)), flush=True)
    xs = np.array([1.0 if o_[i] == o else 0.0 for i, o in cols])
    Am = coo_matrix((Vv, (R, C)), shape=(r, n)).tocsr()
    act = Am @ xs
    viol = [(k, act[k], rlo[k], rhi[k]) for k in range(r) if act[k] > rhi[k] + 1e-6 or act[k] < rlo[k] - 1e-6]
    print("start plan row violations", len(viol), viol[:10], flush=True)
    print("start plan value", V0 + sum(val[o](i) for k, (i, o) in enumerate(cols) if xs[k] > 0.5), flush=True)
A = coo_matrix((Vv, (R, C)), shape=(r, n)).tocsc()
h = highspy.Highs()
h.setOptionValue("output_flag", True)
h.setOptionValue("time_limit", tl)
h.setOptionValue("threads", 1)
h.setOptionValue("presolve", "off")
h.setOptionValue("mip_rel_gap", 0.0)
import os as _os
h.setOptionValue("random_seed", int(_os.environ.get("SEED", "0")))
h.setOptionValue("mip_abs_gap", 0.5)
h.setOptionValue("primal_feasibility_tolerance", 1e-9)
h.setOptionValue("mip_feasibility_tolerance", 1e-9)
lp = highspy.HighsLp()
lp.num_col_ = n; lp.num_row_ = r
lp.col_cost_ = np.array([-float(val[o](i)) for i, o in cols])
lp.col_lower_ = np.zeros(n); lp.col_upper_ = np.ones(n)
lp.row_lower_ = np.array(rlo, float); lp.row_upper_ = np.array(rhi, float)
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
lp.a_matrix_.num_col_ = n; lp.a_matrix_.num_row_ = r
lp.integrality_ = [highspy.HighsVarType.kInteger] * n
h.passModel(lp)
# objective cutoff: we only care about solutions >= T
h.setOptionValue("objective_bound", -(T - V0) + 0.5)
t0 = time.time()
h.run()
info = h.getInfo()
st = h.modelStatusToString(h.getModelStatus())
print("status", st, "time", round(time.time() - t0, 1), flush=True)
if info.primal_solution_status == 2:
    x = np.array(h.getSolution().col_value)
    vsol = V0 + sum(val[o](i) for k, (i, o) in enumerate(cols) if x[k] > 0.5)
    print("best relaxation solution value", vsol, flush=True)
    if outp:
        opt = fixo.copy(); opt[free] = 0
        for k, (i, o) in enumerate(cols):
            if x[k] > 0.5: opt[i] = o
        full = np.nonzero(opt == 1)[0].tolist(); half = np.nonzero(opt == 2)[0].tolist(); src = np.nonzero(opt == 3)[0].tolist()
        srcreq = [[i, "ntic"[t]] for i in half for t in range(4) if I.badt[i, t]]
        json.dump({"full": full, "half": half, "source": src, "srcreq": srcreq, "obj": vsol}, open(outp, "w"))
print("dual bound (value)", V0 - info.mip_dual_bound, flush=True)
print("RESULT T", T, "status", st, flush=True)
