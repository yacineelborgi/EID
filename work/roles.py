"""Role re-assignment MIP with all datasets fixed in place.
Source slots = current source bins (items fixed, fill f). Any mono model of the slot type whose
window contains f can be the source. Targets = models whose every requirement is covered by
a source (value v//2). Data-trained models are kept as they are.
Maximize the total value of targets under the energy cap and per-type coverage balance."""
import json, sys, collections, time
import numpy as np
import highspy
from scipy.sparse import coo_matrix
sys.path.insert(0, '.')
from sol6 import State, TYPES


def solve_roles(st, tlim=300, emax_src=None, verbose=False, gap=0.0, topk=8):
    M, D = st.M, st.D
    cap = st.inst["energyCap"]
    cov = st.covered()
    # pure targets: every req covered by a source
    pure_t = set()
    for m in st.tgt:
        if m in st.used and all((m, t) in cov for t in M[m]["dataRequirements"]):
            pure_t.add(m)
    srcs = list(st.src)
    fixed_models = set(st.used) - set(srcs) - pure_t   # data-trained models (incl. mixed targets)
    fixed_dem = collections.Counter()
    for m in fixed_models:
        for t in M[m]["dataRequirements"]:
            if (m, t) in cov:
                fixed_dem[t] += 1
    E_fixed = sum(M[m]["energyCost"] for m in fixed_models)
    slots = []
    for s in srcs:
        t = next(iter(M[s]["dataRequirements"]))
        f = sum(D[i]["size"] for i in st.bin_items[(s, t)])
        slots.append((s, t, f))
    free_models = [i for i in range(len(M)) if i not in fixed_models]
    monos = collections.defaultdict(list)
    for i in free_models:
        r = M[i]["dataRequirements"]
        if len(r) == 1:
            t, q = next(iter(r.items()))
            if q["upperBound"] >= q["lowerBound"]:
                monos[t].append((q["lowerBound"], q["upperBound"], i))
    # variables
    cols_cost, cols_lb, cols_ub = [], [], []
    rows, colsA, vals = [], [], []
    a_idx = []  # (slot, model, col)
    nc = 0
    for j, (s, t, f) in enumerate(slots):
        cands = [(M[i]["energyCost"], i) for lb, ub, i in monos[t] if lb <= f <= ub]
        cands.sort()
        keepc = set(i for e, i in cands[:topk]) | {s}
        for e, i in cands:
            if i in keepc:
                a_idx.append((j, i, nc)); cols_cost.append(0.0); nc += 1
    z_idx = {}
    for i in free_models:
        v = M[i]["value"] // 2
        if v <= 0:
            continue
        z_idx[i] = nc; cols_cost.append(float(v)); nc += 1
    nr = 0
    rlo, rhi = [], []

    def row(ent, lo, hi):
        nonlocal nr
        for c, v in ent:
            rows.append(nr); colsA.append(c); vals.append(v)
        rlo.append(lo); rhi.append(hi); nr += 1

    by_slot = collections.defaultdict(list); by_model = collections.defaultdict(list)
    for j, i, c in a_idx:
        by_slot[j].append(c); by_model[i].append(c)
    for j in by_slot:
        row([(c, 1) for c in by_slot[j]], -np.inf, 1)
    for i in set(by_model) | set(z_idx):
        ent = [(c, 1) for c in by_model.get(i, [])]
        if i in z_idx:
            ent.append((z_idx[i], 1))
        if len(ent) > 1:
            row(ent, -np.inf, 1)
    for t in TYPES:
        ent = [(c, 1) for j, i, c in a_idx if slots[j][1] == t]
        ent += [(z_idx[i], -1) for i in z_idx if t in M[i]["dataRequirements"]]
        row(ent, fixed_dem[t], np.inf)
    ent = [(c, M[i]["energyCost"] / 1e9) for j, i, c in a_idx] + [(z_idx[i], M[i]["energyCost"] / 1e9) for i in z_idx]
    row(ent, -np.inf, (cap - E_fixed) / 1e9)
    A = coo_matrix((vals, (rows, colsA)), shape=(nr, nc)).tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    h.setOptionValue("time_limit", tlim)
    h.setOptionValue("mip_rel_gap", gap)
    lp = highspy.HighsLp()
    lp.num_col_ = nc; lp.num_row_ = nr
    lp.col_cost_ = np.array(cols_cost); lp.col_lower_ = np.zeros(nc); lp.col_upper_ = np.ones(nc)
    lp.row_lower_ = np.array(rlo, float); lp.row_upper_ = np.array(rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.integrality_ = [highspy.HighsVarType.kInteger] * nc
    h.passModel(lp)
    # hint: current
    x0 = np.zeros(nc)
    for j, i, c in a_idx:
        if slots[j][0] == i:
            x0[c] = 1
    for i in pure_t:
        if i in z_idx:
            x0[z_idx[i]] = 1
    sol = highspy.HighsSolution(); sol.col_value = list(x0); sol.value_valid = True
    h.setSolution(sol)
    h.run()
    x = np.array(h.getSolution().col_value)
    info = h.getInfo()
    cur_val = sum(M[i]["value"] // 2 for i in pure_t)
    return dict(x=x, a_idx=a_idx, z_idx=z_idx, slots=slots, obj=info.objective_function_value,
                bound=info.mip_dual_bound, cur=cur_val, fixed_models=fixed_models, pure_t=pure_t, nvars=nc)


def apply_roles(st, res):
    M = st.M
    x = res["x"]; slots = res["slots"]
    # clear old sources/targets
    old_items = {}
    for j, (s, t, f) in enumerate(slots):
        old_items[j] = list(st.bin_items[(s, t)])
        for i in old_items[j]:
            del st.item_bin[i]
        st.bin_items[(s, t)] = set()
        st.used.discard(s)
    for m in res["pure_t"]:
        st.used.discard(m)
    fixed_pairs = [(s, tg) for s, tg in st.src.items() if tg in res["fixed_models"]]
    st.src = {}
    new_src = collections.defaultdict(list)
    for j, i, c in res["a_idx"]:
        if x[c] > 0.5:
            s, t, f = slots[j]
            for d in old_items[j]:
                st.item_bin[d] = (i, t); st.bin_items[(i, t)].add(d)
            st.used.add(i)
            new_src[t].append(i)
    demand = collections.defaultdict(list)
    for s, tg in fixed_pairs:
        demand[next(iter(M[s]["dataRequirements"]))].append(tg)
    for i, c in res["z_idx"].items():
        if x[c] > 0.5:
            st.used.add(i)
            for t in M[i]["dataRequirements"]:
                demand[t].append(i)
    for t in TYPES:
        assert len(new_src[t]) >= len(demand[t]), (t, len(new_src[t]), len(demand[t]))
        for s, tg in zip(new_src[t], demand[t]):
            st.src[s] = tg
        for s in new_src[t][len(demand[t]):]:
            for d in list(st.bin_items[(s, t)]):
                del st.item_bin[d]
            st.bin_items[(s, t)] = set()
            st.used.discard(s)
    st.tgt = collections.defaultdict(list)
    for s, tg in st.src.items():
        st.tgt[tg].append(s)


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    st = State(inst, sol)
    t0 = time.time()
    res = solve_roles(st, tlim=float(sys.argv[4]), verbose=False)
    print("vars", res["nvars"], "obj", res["obj"], "bound", res["bound"], "cur", res["cur"], "time", round(time.time() - t0), flush=True)
    apply_roles(st, res)
    sc = st.score()
    print(sc)
    if sc[1]:
        json.dump(st.to_sol(), open(sys.argv[3], "w"))
