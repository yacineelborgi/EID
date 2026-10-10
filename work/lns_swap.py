"""Model-selection LNS for 6_shortage-like instances.
Each iteration: pick removable used models (sources, mono half targets, full/half monos)
of a few types; CP-SAT decides which to drop and which unused mono models to add
(as sources, half targets, full or half data models), with exact item packing from the
pool + items of dropped models, energy limited. Sources are interchangeable inside a type:
coverage supply >= demand per type."""
import json, sys, random, time, collections
from ortools.sat.python import cp_model
sys.path.insert(0, '.')
from sol6 import State, TYPES


def mono_type(M, m):
    r = M[m]["dataRequirements"]
    return next(iter(r)) if len(r) == 1 else None


def iterate(st, rng, types, n_src=40, n_tgt=30, n_mono=20, n_new_src=60, n_new_tgt=60, n_new_data=40,
            tlim=20, workers=4, log=True):
    M, D = st.M, st.D
    cap = st.inst["energyCap"]
    ES = cap - sum(M[m]["energyCost"] for m in st.used)
    full = {m for m in st.used if m not in st.src and st.is_full(m)}
    cov = st.covered()
    # current per-type supply/demand of coverage
    supply = collections.Counter(mono_type(M, s) for s in st.src)
    demand = collections.Counter(next(iter(M[s]["dataRequirements"])) for s in st.src)  # equal now
    rem = []  # (model, kind)
    for t in types:
        srcs = [s for s in st.src if mono_type(M, s) == t]
        tg = [m for m in st.used if m not in st.src and len(M[m]["dataRequirements"]) == 1 and t in M[m]["dataRequirements"] and st.tgt.get(m)]
        mono_data = [m for m in st.used if m not in st.src and not st.tgt.get(m) and len(M[m]["dataRequirements"]) == 1 and t in M[m]["dataRequirements"]]
        # bias sources toward high energy
        srcs.sort(key=lambda s: -M[s]["energyCost"])
        pick = srcs[:n_src // 2] + rng.sample(srcs[n_src // 2:], min(n_src - n_src // 2, max(0, len(srcs) - n_src // 2)))
        rem += [(s, "src") for s in pick]
        rem += [(m, "tgt") for m in rng.sample(tg, min(n_tgt, len(tg)))]
        rem += [(m, "data") for m in rng.sample(mono_data, min(n_mono, len(mono_data)))]
    unused = [i for i in range(len(M)) if i not in st.used]
    mdl = cp_model.CpModel()
    keep = {}
    obj, energy = [], []
    sup = collections.defaultdict(list)
    dem = collections.defaultdict(list)
    avail = collections.defaultdict(list)  # type -> items available (pool + items of removable)
    owner = {}
    for t in types:
        for i in range(len(D)):
            pass
    pool = collections.defaultdict(list)
    for d in D:
        if d["dataType"] in types and d["id"] not in st.item_bin:
            pool[d["dataType"]].append(d["id"])
    base_obj = 0
    for (m, kind) in rem:
        k = mono_type(M, m)
        v = mdl.NewBoolVar("")
        keep[m] = v
        energy.append(M[m]["energyCost"] * (v - 1))  # removing frees energy
        if kind == "src":
            sup[k].append(v)
        elif kind == "tgt":
            dem[k].append(v)
            obj.append((M[m]["value"] // 2) * (v - 1))
        else:
            val = M[m]["value"] if m in full else M[m]["value"] // 2
            obj.append(val * (v - 1))
        for i in st.bin_items.get((m, k), ()):
            owner[i] = m
            avail[k].append(i)
    remset = set(m for m, _ in rem)
    x = collections.defaultdict(dict)
    newv = {}
    for t in types:
        items = pool[t] + avail[t]
        vol = sum(D[i]["size"] for i in items)
        fvol = sum(D[i]["size"] for i in items if not D[i]["isCopyrighted"])
        monos = [i for i in unused if mono_type(M, i) == t]
        good = [i for i in monos if M[i]["dataRequirements"][t]["upperBound"] >= M[i]["dataRequirements"][t]["lowerBound"]]
        cs = sorted([i for i in good if M[i]["dataRequirements"][t]["lowerBound"] <= vol],
                    key=lambda i: (M[i]["dataRequirements"][t]["lowerBound"] + M[i]["energyCost"] / 2e7))[:n_new_src]
        ct = sorted(monos, key=lambda i: -(M[i]["value"] // 2) / (M[i]["energyCost"] + 1e9))[:n_new_tgt]
        cf = sorted([i for i in good if M[i]["dataRequirements"][t]["lowerBound"] <= fvol],
                    key=lambda i: -M[i]["value"] / (M[i]["energyCost"] + 1e9 + 2e7 * M[i]["dataRequirements"][t]["lowerBound"]))[:n_new_data]
        roles = collections.defaultdict(list)
        for m in ct:
            if M[m]["value"] // 2 <= 0:
                continue
            v = mdl.NewBoolVar(""); roles[m].append(v); newv[(m, "tgt")] = v
            dem[t].append(v); obj.append((M[m]["value"] // 2) * v); energy.append(M[m]["energyCost"] * v)
        for mode, lst in (("src", cs), ("full", cf), ("half", cf)):
            for m in lst:
                r = M[m]["dataRequirements"][t]; lb, ub = r["lowerBound"], r["upperBound"]
                y = mdl.NewBoolVar(""); roles[m].append(y); newv[(m, mode)] = y
                key = (m, mode)
                for i in items:
                    if D[i]["size"] <= ub and (mode != "full" or not D[i]["isCopyrighted"]):
                        xv = mdl.NewBoolVar("")
                        x[key][i] = xv
                        if i in owner:
                            mdl.AddImplication(xv, keep[owner[i]].Not())
                e = sum(D[i]["size"] * xv for i, xv in x[key].items())
                mdl.Add(e >= lb).OnlyEnforceIf(y)
                mdl.Add(e <= ub * y)
                energy.append(M[m]["energyCost"] * y)
                if mode == "src":
                    sup[t].append(y)
                elif mode == "full":
                    obj.append(M[m]["value"] * y)
                else:
                    obj.append((M[m]["value"] // 2) * y)
        for m, vs in roles.items():
            if len(vs) > 1:
                mdl.AddAtMostOne(vs)
    byi = collections.defaultdict(list)
    for key, d in x.items():
        for i, v in d.items():
            byi[i].append(v)
    for i, vs in byi.items():
        mdl.AddAtMostOne(vs)
    # coverage balance per type: kept sources + new sources >= kept targets + new targets + fixed demand
    for t in types:
        fixed_sup = sum(1 for s in st.src if mono_type(M, s) == t and s not in remset)
        fixed_dem = sum(1 for s in st.src if mono_type(M, s) == t) - sum(1 for (m, kind) in rem if kind == "tgt" and mono_type(M, m) == t)
        mdl.Add(fixed_sup + sum(sup[t]) >= fixed_dem + sum(dem[t]))
    mdl.Add(sum(energy) <= ES)
    mdl.Maximize(sum(obj))
    for m, v in keep.items():
        mdl.AddHint(v, 1)
    for key, v in newv.items():
        mdl.AddHint(v, 0)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tlim
    sv.parameters.num_workers = workers
    sv.parameters.random_seed = rng.randrange(10 ** 6)
    r = sv.Solve(mdl)
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 0
    gain = round(sv.ObjectiveValue())
    if log:
        print(f"   lns {''.join(types)} gain {gain} {sv.StatusName(r)} bound {sv.BestObjectiveBound():.0f}", flush=True)
    if gain <= 0:
        return 0
    # ---- apply ----
    dropped = set(m for m, v in keep.items() if not sv.Value(v))
    old_pairs = {t: [(s, tg) for s, tg in st.src.items() if mono_type(M, s) == t] for t in types}
    for m in dropped:
        k = mono_type(M, m)
        for i in list(st.bin_items.get((m, k), ())):
            del st.item_bin[i]
        st.bin_items[(m, k)] = set()
        st.used.discard(m)
    new_src, new_tgt = collections.defaultdict(list), collections.defaultdict(list)
    for (m, mode), y in newv.items():
        if not sv.Value(y):
            continue
        t = mono_type(M, m)
        st.used.add(m)
        if mode == "tgt":
            new_tgt[t].append(m)
            continue
        for i, xv in x[(m, mode)].items():
            if sv.Value(xv):
                st.item_bin[i] = (m, t); st.bin_items[(m, t)].add(i)
        if mode == "src":
            new_src[t].append(m)
    for t in types:
        demand = [tg for s, tg in old_pairs[t] if tg not in dropped] + new_tgt[t]
        supply = [s for s, tg in old_pairs[t] if s not in dropped] + new_src[t]
        for s, tg in old_pairs[t]:
            del st.src[s]
        assert len(supply) >= len(demand), (t, len(supply), len(demand))
        for s, tg in zip(supply, demand):
            st.src[s] = tg
        for s in supply[len(demand):]:
            for i in list(st.bin_items.get((s, t), ())):
                del st.item_bin[i]
            st.bin_items[(s, t)] = set()
            st.used.discard(s)
    st.tgt = collections.defaultdict(list)
    for s, tg in st.src.items():
        st.tgt[tg].append(s)
    return gain


def main():
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    out = sys.argv[3]
    iters = int(sys.argv[4]); seed = int(sys.argv[5]); tlim = float(sys.argv[6])
    st = State(inst, sol)
    rng = random.Random(seed)
    best = st.score()[0]
    print("start", best, flush=True)
    for it in range(iters):
        types = rng.sample(TYPES, rng.choice([1, 2]))
        snap = (dict(st.src), dict(st.item_bin), set(st.used))
        g = iterate(st, rng, types, tlim=tlim)
        if g > 0:
            sc, ok, msg = st.score()
            if ok and sc > best:
                best = sc
                json.dump(st.to_sol(), open(out, "w"))
                print(f"it {it} NEW BEST {best}", flush=True)
            else:
                print(f"it {it} rejected ({sc}, {ok}, {msg})", flush=True)
                st.src, st.item_bin, st.used = snap
                st.bin_items = collections.defaultdict(set)
                for d, b in st.item_bin.items():
                    st.bin_items[b].add(d)
                st.tgt = collections.defaultdict(list)
                for s, tg in st.src.items():
                    st.tgt[tg].append(s)


if __name__ == "__main__":
    main()
