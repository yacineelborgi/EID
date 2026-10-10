"""Sink LNS: grow new 'sink' bins (new sources / new full bins) by re-packing groups of
existing bins of the same type with CP-SAT so that their waste flows into the sinks.
usage: python lns_sink.py inst sol type iters K tlim seed out sinkspec...
sinkspec = model:freeonly(0/1)"""
import json, sys, random, time, collections
from ortools.sat.python import cp_model
sys.path.insert(0, '.')
from sol6 import State


def run(st, t, sinks, iters, K, tlim, seed, log=True, ntop=200):
    """sinks: list of (key, lb, ub, freeonly). Their contents live in st.bin_items too."""
    rng = random.Random(seed)
    M, D = st.M, st.D
    full = {m for m in st.used if m not in st.src and st.is_full(m)}
    cov = st.covered()
    bins = []
    for m in st.used:
        r = M[m]["dataRequirements"].get(t)
        if not r:
            continue
        if m not in st.src and (m, t) in cov:
            continue
        bins.append(((m, t), r["lowerBound"], r["upperBound"], m in full))
    sinkkeys = set(s[0] for s in sinks)
    bins = [b for b in bins if b[0] not in sinkkeys]
    t0 = time.time()
    for it in range(iters):
        fills = {s[0]: sum(D[i]["size"] for i in st.bin_items[s[0]]) for s in sinks}
        if all(fills[s[0]] >= s[1] for s in sinks):
            break
        waste = {b[0]: sum(D[i]["size"] for i in st.bin_items[b[0]]) - b[1] for b in bins}
        bs = sorted(bins, key=lambda b: -waste[b[0]])
        top, rest = bs[:ntop], bs[ntop:]
        rng.shuffle(top); rng.shuffle(rest)
        sel = top[:K // 2] + rest[:K - K // 2]
        items = set()
        for b in sel:
            items |= st.bin_items[b[0]]
        for s in sinks:
            items |= st.bin_items[s[0]]
        pool = [d["id"] for d in D if d["dataType"] == t and d["id"] not in st.item_bin]
        items |= set(pool)
        items = sorted(items)
        mdl = cp_model.CpModel()
        x = {}
        allb = sel + list(sinks)
        for bi, (k, lb, ub, fo) in enumerate(allb):
            for i in items:
                if D[i]["size"] > ub or (fo and D[i]["isCopyrighted"]):
                    continue
                x[i, bi] = mdl.NewBoolVar("")
        byi = collections.defaultdict(list); byb = collections.defaultdict(list)
        for (i, bi), v in x.items():
            byi[i].append(v); byb[bi].append((i, v))
        for i, vs in byi.items():
            mdl.AddAtMostOne(vs)
        obj = []
        for bi, (k, lb, ub, fo) in enumerate(allb):
            e = sum(D[i]["size"] * v for i, v in byb[bi])
            if bi < len(sel):
                mdl.Add(e >= lb); mdl.Add(e <= ub)
            else:
                mdl.Add(e <= ub)
                f = mdl.NewIntVar(0, lb, "")
                mdl.Add(f <= e)
                obj.append(f)
        used_sel = sum(D[i]["size"] * v for (i, bi), v in x.items() if bi < len(sel))
        mdl.Maximize(100000 * sum(obj) - used_sel)
        for (i, bi), v in x.items():
            mdl.AddHint(v, 1 if st.item_bin.get(i) == allb[bi][0] else 0)
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = tlim
        sv.parameters.num_workers = 1
        sv.parameters.random_seed = rng.randrange(10 ** 6)
        r = sv.Solve(mdl)
        if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            continue
        before = sum(min(fills[s[0]], s[1]) for s in sinks)
        after = sum(sv.Value(f) for f in obj)
        old_used = sum(D[i]["size"] for b in sel for i in st.bin_items[b[0]])
        if (after, -sv.Value(used_sel)) > (before, -old_used):
            for b in allb:
                for i in st.bin_items[b[0]]:
                    del st.item_bin[i]
                st.bin_items[b[0]] = set()
            for (i, bi), v in x.items():
                if sv.Value(v):
                    st.item_bin[i] = allb[bi][0]
                    st.bin_items[allb[bi][0]].add(i)
        if log:
            pv = sum(d["size"] for d in D if d["dataType"] == t and d["id"] not in st.item_bin)
            print(f"it {it} sinks {before}->{after} need {sum(s[1] for s in sinks)} pool {pv} {time.time()-t0:.0f}s", flush=True)
    return st


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    t = sys.argv[3]
    iters, K, tlim, seed = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
    out = sys.argv[8]
    st = State(inst, sol)
    sinks = []
    for spec in sys.argv[9:]:
        m, fo = spec.split(":")
        m = int(m)
        r = st.M[m]["dataRequirements"][t]
        sinks.append(((m, t), r["lowerBound"], r["upperBound"], fo == "1"))
    run(st, t, sinks, iters, K, tlim, seed)
    fills = {s[0]: sum(st.D[i]["size"] for i in st.bin_items[s[0]]) for s in sinks}
    print("final sink fills", fills, "need", [(s[0], s[1]) for s in sinks])
    json.dump({"dataMappings": [[d, b[0]] for d, b in sorted(st.item_bin.items())],
               "modelMappings": [[s, tt] for s, tt in sorted(st.src.items())]}, open(out, "w"))
