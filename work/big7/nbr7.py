"""Neighbourhood MIP around a plan: free models = those with >1 option within reduced-cost deviation D
(LP duals of lp_state.pkl) or whose plan option is not the best; everything else fixed.
Rows (integer data, exact): energy (in units of 1e6 with integer-checked result), free data per type
(sorted-matching consumption: free models may take `full` only if all their lbs are > TINY, so the
tiny matching of fixed bins is unchanged and consumption changes by exactly sum lb), source balance.
usage: nbr7.py plan.json out.json D tl margins emargin_units
"""
import sys, json, pickle, time
import numpy as np
import highspy
from scipy.sparse import coo_matrix
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
from lp7 import Inst
from mip7 import profits
INF = highspy.kHighsInf
TINY = 400


def match_lb(I, full, t):
    full = np.asarray(full, int)
    m = full[I.has[full, t]]
    l = np.sort(I.lb[m, t])
    return int(np.maximum(l, I.fs[t][:len(l)]).sum())


def main():
    plan = json.load(open(sys.argv[1])); out = sys.argv[2]
    D = float(sys.argv[3]); tl = float(sys.argv[4])
    margin = [int(x) for x in sys.argv[5].split(",")]
    emarg = int(sys.argv[6])  # energy margin (absolute units)
    I = Inst(); nM = I.nM
    cuts, y, obj = pickle.load(open("/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7/lp_state.pkl", "rb"))
    P = profits(I, y); best = P.max(1); dev = best[:, None] - P
    opt = np.zeros(nM, int); opt[plan["full"]] = 1; opt[plan["half"]] = 2; opt[plan["source"]] = 3
    allow = (dev <= D) & np.isfinite(dev)
    allow[np.arange(nM), opt] = True
    allow[:, 0] = True  # switching off is always allowed (dev may be large but it's a legal option)
    allow[:, 0] = dev[:, 0] <= D
    allow[np.arange(nM), opt] = True
    tiny = ((I.lb <= TINY) & I.has).any(1)
    allow[tiny, 1] = (opt[tiny] == 1)  # tiny models keep their full status
    allow[tiny & (opt == 1), :] = False; allow[tiny & (opt == 1), 1] = True
    free = np.nonzero(allow.sum(1) > 1)[0]
    print("free models", len(free), flush=True)
    isfree = np.zeros(nM, bool); isfree[free] = True
    fixed_used = np.nonzero((~isfree) & (opt > 0))[0]
    E_fixed = int(sum(int(I.e[i]) for i in fixed_used))
    full_fixed = [i for i in plan["full"] if not isfree[i]]
    # data: consumption of fixed full bins via matching (free full models have no tiny bins)
    Cfix = [match_lb(I, full_fixed, t) for t in range(4)]
    # variables: for each free model and allowed option o in 1..3, a binary
    cols = []  # (i, o)
    for i in free:
        for o in (1, 2, 3):
            if allow[i, o]:
                cols.append((int(i), o))
    n = len(cols)
    vals = {1: lambda i: int(I.v[i]), 2: lambda i: int(I.v[i]) // 2, 3: lambda i: 0}
    c = np.array([-vals[o](i) for i, o in cols], float)
    R, C, V, rlo, rhi = [], [], [], [], []
    r = 0
    # one option per model
    pos = {}
    for k, (i, o) in enumerate(cols):
        pos.setdefault(i, []).append(k)
    for i, ks in pos.items():
        for k in ks:
            R.append(r); C.append(k); V.append(1.0)
        rlo.append(-INF); rhi.append(1.0); r += 1
    # energy (units of 1e6, coefficient rounded up -> conservative)
    SC = 1000
    rE = r
    for k, (i, o) in enumerate(cols):
        R.append(r); C.append(k); V.append(float(-(-int(I.e[i]) // SC)))
    rhsE = (I.cap - E_fixed - emarg) // SC
    rlo.append(-INF); rhi.append(float(rhsE)); r += 1
    # free data
    for t in range(4):
        for k, (i, o) in enumerate(cols):
            if o == 1 and I.has[i, t]:
                R.append(r); C.append(k); V.append(float(I.lb[i, t]))
        rlo.append(-INF); rhi.append(float(I.F[t] - Cfix[t] - margin[t])); r += 1
    # sources: bad reqs of half (non-bad reqs are always filled with data here) <= sources
    for t in range(4):
        fixed_need = sum(1 for i in plan["half"] if not isfree[i] and I.badt[i, t])
        fixed_src = sum(1 for i in plan["source"] if not isfree[i] and I.mtype[i] == t)
        for k, (i, o) in enumerate(cols):
            if o == 2 and I.badt[i, t]:
                R.append(r); C.append(k); V.append(1.0)
            if o == 3 and I.mtype[i] == t:
                R.append(r); C.append(k); V.append(-1.0)
        rlo.append(-INF); rhi.append(float(fixed_src - fixed_need)); r += 1
    A = coo_matrix((V, (R, C)), shape=(r, n)).tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", True)
    h.setOptionValue("time_limit", tl)
    h.setOptionValue("threads", 1)
    h.setOptionValue("mip_rel_gap", 0.0)
    import os
    h.setOptionValue("random_seed", int(os.environ.get("SEED", "0")))
    h.setOptionValue("mip_abs_gap", 0.99)
    h.setOptionValue("primal_feasibility_tolerance", 1e-9)
    h.setOptionValue("mip_feasibility_tolerance", 1e-9)
    lp = highspy.HighsLp()
    lp.num_col_ = n; lp.num_row_ = r
    lp.col_cost_ = c; lp.col_lower_ = np.zeros(n); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.array(rlo, float); lp.row_upper_ = np.array(rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.a_matrix_.num_col_ = n; lp.a_matrix_.num_row_ = r
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    h.passModel(lp)
    st = np.array([1.0 if opt[i] == o else 0.0 for i, o in cols])
    s = highspy.HighsSolution(); s.col_value = list(st); s.value_valid = True
    h.setSolution(s)
    base_val = int(sum(int(I.v[i]) for i in plan["full"] if not isfree[i]) + sum(int(I.v[i]) // 2 for i in plan["half"] if not isfree[i]))
    print("start value", base_val - c @ st, flush=True)
    h.run()
    x = np.array(h.getSolution().col_value)
    info = h.getInfo()
    print("MIP obj", base_val - info.objective_function_value, "bound", base_val - info.mip_dual_bound, flush=True)
    newopt = opt.copy()
    newopt[free] = 0
    for k, (i, o) in enumerate(cols):
        if x[k] > 0.5:
            newopt[i] = o
    full = np.nonzero(newopt == 1)[0].tolist(); half = np.nonzero(newopt == 2)[0].tolist(); src = np.nonzero(newopt == 3)[0].tolist()
    TY = "ntic"
    srcreq = [[i, TY[t]] for i in half for t in range(4) if I.badt[i, t]]
    p = {"full": full, "half": half, "source": src, "srcreq": srcreq}
    val = int(I.v[full].sum() + (I.v[half] // 2).sum())
    E = int(sum(int(I.e[i]) for i in full + half + src))
    print("plan value", val, "energy slack", I.cap - E, flush=True)
    for t in range(4):
        print("  type", t, "free slack (matching)", I.F[t] - match_lb(I, full, t),
              "src need", sum(1 for i in half if I.badt[i, t]), "have", sum(1 for i in src if I.mtype[i] == t), flush=True)
    print("changes", int((newopt != opt).sum()), flush=True)
    p["obj"] = val
    json.dump(p, open(out, "w"))


main()
