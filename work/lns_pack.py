"""Packing LNS: re-pack random groups of bins of one type with CP-SAT, minimizing the
volume used, so that waste is pushed out into the pool of unused datasets."""
import json, sys, random, time, collections
from ortools.sat.python import cp_model
sys.path.insert(0, '.')
from sol6 import State


def bins_of_type(st, t, full_set):
    cov = st.covered()
    out = []
    for m in st.used:
        if m in st.src:
            r = st.M[m]["dataRequirements"].get(t)
            if r:
                out.append(((m, t), r["lowerBound"], r["upperBound"], False))
            continue
        r = st.M[m]["dataRequirements"].get(t)
        if not r or (m, t) in cov:
            continue
        out.append(((m, t), r["lowerBound"], r["upperBound"], m in full_set))
    return out


def solve_sub(st, bins, items, tlim=5.0, hint=True, workers=1):
    D = st.D
    mdl = cp_model.CpModel()
    x = {}
    for b, (k, lb, ub, fo) in enumerate(bins):
        for i in items:
            s = D[i]["size"]
            if s > ub or (fo and D[i]["isCopyrighted"]):
                continue
            x[i, b] = mdl.NewBoolVar("")
    byi = collections.defaultdict(list)
    byb = collections.defaultdict(list)
    for (i, b), v in x.items():
        byi[i].append(v); byb[b].append((i, v))
    for i, vs in byi.items():
        mdl.AddAtMostOne(vs)
    for b, (k, lb, ub, fo) in enumerate(bins):
        e = sum(D[i]["size"] * v for i, v in byb[b])
        mdl.Add(e >= lb); mdl.Add(e <= ub)
    mdl.Minimize(sum(D[i]["size"] * v for (i, b), v in x.items()))
    if hint:
        for (i, b), v in x.items():
            mdl.AddHint(v, 1 if st.item_bin.get(i) == bins[b][0] else 0)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tlim
    sv.parameters.num_workers = workers
    r = sv.Solve(mdl)
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    asg = collections.defaultdict(list)
    for (i, b), v in x.items():
        if sv.Value(v):
            asg[bins[b][0]].append(i)
    return asg, sv.ObjectiveValue(), r == cp_model.OPTIMAL


def run(st, iters=200, K=30, tlim=3.0, seed=0, types="ntic", log=True, maxpool=200):
    rng = random.Random(seed)
    D = st.D
    full_set = {m for m in st.used if m not in st.src and st.is_full(m)}
    t0 = time.time()
    for it in range(iters):
        t = rng.choice(types)
        bins = bins_of_type(st, t, full_set)
        # bias: half random, half bins with waste
        waste = {b[0]: sum(D[i]["size"] for i in st.bin_items[b[0]]) - b[1] for b in bins}
        bins.sort(key=lambda b: -waste[b[0]])
        top = bins[:150]
        rest = bins[150:]
        rng.shuffle(top); rng.shuffle(rest)
        sel = top[:K // 2] + rest[:K - K // 2]
        items = set()
        for b in sel:
            items |= st.bin_items[b[0]]
        pool = [d["id"] for d in D if d["dataType"] == t and d["id"] not in st.item_bin]
        rng.shuffle(pool)
        items |= set(pool[:maxpool])
        before = sum(D[i]["size"] for b in sel for i in st.bin_items[b[0]])
        res = solve_sub(st, sel, sorted(items), tlim)
        if res is None:
            continue
        asg, obj, opt = res
        if obj < before:
            for b in sel:
                for i in st.bin_items[b[0]]:
                    del st.item_bin[i]
                st.bin_items[b[0]] = set()
            for k, its in asg.items():
                for i in its:
                    st.item_bin[i] = k
                st.bin_items[k] = set(its)
        if log and (it % 10 == 0 or obj < before):
            pv = {tt: sum(d["size"] for d in D if d["dataType"] == tt and d["id"] not in st.item_bin) for tt in "ntic"}
            print(f"it {it} t {t} before {before} after {obj:.0f} opt {opt} pool {pv} {time.time()-t0:.0f}s", flush=True)
    return st


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    st = State(inst, sol)
    run(st, iters=int(sys.argv[3]), K=int(sys.argv[4]), tlim=float(sys.argv[5]), seed=int(sys.argv[6]), types=sys.argv[8])
    print(st.score())
    json.dump(st.to_sol(), open(sys.argv[7], "w"))
