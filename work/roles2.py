"""Extended role re-assignment (datasets fixed in place).
Slots = every mono-type bin currently filled (source bins and mono data bins), with fill f
and a flag 'all items free'. A slot hosts at most one mono model whose window contains f:
  - as a source  (gives one coverage of the slot type, value 0)
  - as an owner  (value v if all items free else v//2)
Targets = models whose every requirement is covered by a source (value v//2).
Multi-type data-trained models are fixed. Maximize total value under energy and coverage."""
import json, sys, collections, time
import numpy as np
import highspy
from scipy.sparse import coo_matrix
sys.path.insert(0, '.')
from sol6 import State, TYPES


def solve_roles2(st, tlim=600, verbose=False, gap=0.0, threads=1):
    M, D = st.M, st.D
    cap = st.inst["energyCap"]
    cov = st.covered()
    mono = lambda m: len(M[m]["dataRequirements"]) == 1
    pure_t = set(m for m in st.tgt if m in st.used and all((m, t) in cov for t in M[m]["dataRequirements"]))
    slot_models = set(m for m in st.used if mono(m) and m not in pure_t)  # sources + mono data models
    fixed_models = set(st.used) - slot_models - pure_t
    fixed_dem = collections.Counter()
    for m in fixed_models:
        for t in M[m]["dataRequirements"]:
            if (m, t) in cov:
                fixed_dem[t] += 1
    E_fixed = sum(M[m]["energyCost"] for m in fixed_models)
    slots = []
    for m in slot_models:
        t = next(iter(M[m]["dataRequirements"]))
        its = st.bin_items[(m, t)]
        f = sum(D[i]["size"] for i in its)
        fr = all(not D[i]["isCopyrighted"] for i in its)
        slots.append((m, t, f, fr))
    cand_models = [i for i in range(len(M)) if i not in fixed_models]
    monos = collections.defaultdict(list)
    for i in cand_models:
        r = M[i]["dataRequirements"]
        if len(r) == 1:
            t, q = next(iter(r.items()))
            if q["upperBound"] >= q["lowerBound"]:
                monos[t].append((q["lowerBound"], q["upperBound"], i))
    for t in monos:
        monos[t].sort()
    import bisect
    cost, cols = [], []  # col descriptors
    nc = 0
    a_src, a_own = [], []
    for j, (m0, t, f, fr) in enumerate(slots):
        lst = monos[t]
        hi = bisect.bisect_right(lst, (f, 10 ** 12, 10 ** 12))
        for lb, ub, i in lst[:hi]:
            if ub < f:
                continue
            a_src.append((j, i, nc)); cost.append(0.0); nc += 1
            val = M[i]["value"] if fr else M[i]["value"] // 2
            if val > 0:
                a_own.append((j, i, nc)); cost.append(float(val)); nc += 1
    z_idx = {}
    for i in cand_models:
        v = M[i]["value"] // 2
        if v > 0:
            z_idx[i] = nc; cost.append(float(v)); nc += 1
    rows, colsA, vals, rlo, rhi = [], [], [], [], []
    nr = 0

    def row(ent, lo, hi):
        nonlocal nr
        for c, v in ent:
            rows.append(nr); colsA.append(c); vals.append(v)
        rlo.append(lo); rhi.append(hi); nr += 1

    by_slot = collections.defaultdict(list); by_model = collections.defaultdict(list)
    for j, i, c in a_src + a_own:
        by_slot[j].append(c); by_model[i].append(c)
    for i, c in z_idx.items():
        by_model[i].append(c)
    for j, cs in by_slot.items():
        row([(c, 1) for c in cs], -np.inf, 1)
    for i, cs in by_model.items():
        if len(cs) > 1:
            row([(c, 1) for c in cs], -np.inf, 1)
    for t in TYPES:
        ent = [(c, 1) for j, i, c in a_src if slots[j][1] == t]
        ent += [(z_idx[i], -1) for i in z_idx if t in M[i]["dataRequirements"]]
        row(ent, fixed_dem[t], np.inf)
    ent = [(c, M[i]["energyCost"] / 1e9) for j, i, c in a_src + a_own] + [(c, M[i]["energyCost"] / 1e9) for i, c in z_idx.items()]
    row(ent, -np.inf, (cap - E_fixed) / 1e9)
    A = coo_matrix((vals, (rows, colsA)), shape=(nr, nc)).tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    h.setOptionValue("time_limit", tlim)
    h.setOptionValue("mip_rel_gap", gap)
    h.setOptionValue("threads", threads)
    lp = highspy.HighsLp()
    lp.num_col_ = nc; lp.num_row_ = nr
    lp.col_cost_ = np.array(cost); lp.col_lower_ = np.zeros(nc); lp.col_upper_ = np.ones(nc)
    lp.row_lower_ = np.array(rlo, float); lp.row_upper_ = np.array(rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.integrality_ = [highspy.HighsVarType.kInteger] * nc
    h.passModel(lp)
    x0 = np.zeros(nc)
    srcset = set(st.src)
    for j, i, c in a_src:
        if slots[j][0] == i and i in srcset:
            x0[c] = 1
    for j, i, c in a_own:
        if slots[j][0] == i and i not in srcset:
            x0[c] = 1
    for i in pure_t:
        if i in z_idx:
            x0[z_idx[i]] = 1
    cur_val = sum(x0[c] * cost[c] for c in range(nc))
    sol = highspy.HighsSolution(); sol.col_value = list(x0); sol.value_valid = True
    h.setSolution(sol)
    h.run()
    x = np.array(h.getSolution().col_value)
    info = h.getInfo()
    return dict(x=x, a_src=a_src, a_own=a_own, z_idx=z_idx, slots=slots, obj=info.objective_function_value,
                bound=info.mip_dual_bound, cur=cur_val, fixed_models=fixed_models, nvars=nc)


def apply_roles2(st, res):
    M = st.M
    x = res["x"]; slots = res["slots"]
    old_items = {}
    for j, (m0, t, f, fr) in enumerate(slots):
        old_items[j] = list(st.bin_items[(m0, t)])
        for d in old_items[j]:
            del st.item_bin[d]
        st.bin_items[(m0, t)] = set()
    keep = set(res["fixed_models"])
    fixed_pairs = [(s, tg) for s, tg in st.src.items() if tg in keep]
    st.used = set(keep)
    st.src = {}
    new_src = collections.defaultdict(list)
    for kind, lst in (("src", res["a_src"]), ("own", res["a_own"])):
        for j, i, c in lst:
            if x[c] > 0.5:
                m0, t, f, fr = slots[j]
                for d in old_items[j]:
                    st.item_bin[d] = (i, t); st.bin_items[(i, t)].add(d)
                st.used.add(i)
                if kind == "src":
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
    res = solve_roles2(st, tlim=float(sys.argv[4]), threads=int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    print("vars", res["nvars"], "obj", res["obj"], "bound", res["bound"], "cur", res["cur"], "time", round(time.time() - t0), flush=True)
    apply_roles2(st, res)
    sc = st.score()
    print(sc)
    if sc[1]:
        json.dump(st.to_sol(), open(sys.argv[3], "w"))
