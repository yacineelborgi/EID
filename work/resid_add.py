"""Use the pool (unused datasets) and the spare energy to add models, exactly (CP-SAT).
Candidates per type t (all currently unused models):
  - mono models as new sources (filled from pool items)
  - mono models as half targets covered by a new source (value v//2, no data)
  - mono models trained in full mode from free pool items (value v)
  - mono models trained in half mode from any pool items (value v//2)
Optionally a set of currently used models can be removed (freed energy + items)."""
import json, sys, collections, bisect
from ortools.sat.python import cp_model
sys.path.insert(0, '.')
from sol6 import State, TYPES


def add_models(st, tlim=60, emax_src=3e11, max_tgt=300, max_full=300, workers=4, log=True, multi_cands=(), extra_mono=(), hint=None, cap_src=120, cap_data=120):
    M, D = st.M, st.D
    cap = st.inst["energyCap"]
    E_used = 0
    for m in st.used:
        E_used += M[m]["energyCost"]
    ES = cap - E_used
    pool = {t: [d["id"] for d in D if d["dataType"] == t and d["id"] not in st.item_bin] for t in TYPES}
    unused = [i for i in range(len(M)) if i not in st.used]
    mdl = cp_model.CpModel()
    obj = []
    energy = []
    x = collections.defaultdict(dict)  # bin key -> item -> var
    ybin = {}
    zt = {}
    nsrc = collections.defaultdict(list)
    ntgt = collections.defaultdict(list)
    roles = collections.defaultdict(list)  # model -> list of role vars
    for t in TYPES:
        pv = sum(D[i]["size"] for i in pool[t])
        pf = sum(D[i]["size"] for i in pool[t] if not D[i]["isCopyrighted"])
        monos = [i for i in unused if len(M[i]["dataRequirements"]) == 1 and t in M[i]["dataRequirements"]]
        # targets: best value per energy (+ forced extra candidates)
        tg = sorted(monos, key=lambda i: -(M[i]["value"] // 2) / (M[i]["energyCost"] + 1))[:max_tgt]
        extra_t = [i for i in extra_mono if i in set(monos)]
        tg = list(dict.fromkeys(tg + extra_t))
        for m in tg:
            if M[m]["value"] // 2 <= 0 or M[m]["energyCost"] > ES:
                continue
            v = mdl.NewBoolVar(""); zt[m] = v; roles[m].append(v)
            ntgt[t].append(v); obj.append((M[m]["value"] // 2) * v); energy.append(M[m]["energyCost"] * v)
        okm = []
        for m in monos:
            r = M[m]["dataRequirements"][t]
            lb, ub = r["lowerBound"], r["upperBound"]
            if ub < lb or lb > pv or M[m]["energyCost"] > ES:
                continue
            okm.append(m)
        forced = set(extra_mono) | set(hint or ())
        LAM, MU = 2.8e-10, 0.0052
        lbof = lambda m: M[m]["dataRequirements"][t]["lowerBound"]
        top_src = set(sorted((m for m in okm if M[m]["energyCost"] <= emax_src),
                             key=lambda m: LAM * M[m]["energyCost"] + MU * lbof(m))[:cap_src])
        top_full = set(sorted((m for m in okm if lbof(m) <= pf),
                              key=lambda m: -(M[m]["value"] - LAM * M[m]["energyCost"] - MU * lbof(m)))[:cap_data])
        top_half = set(sorted(okm, key=lambda m: -(M[m]["value"] // 2 - LAM * M[m]["energyCost"] - MU * lbof(m)))[:cap_data // 2])
        for m in okm:
            r = M[m]["dataRequirements"][t]
            lb, ub = r["lowerBound"], r["upperBound"]
            modes = []
            if M[m]["energyCost"] <= emax_src and (m in top_src or m in forced):
                modes.append("src")
            if lb <= pf and (m in top_full or m in forced):
                modes.append("full")
            if m in top_half or m in forced:
                modes.append("half")
            for mode in modes:
                key = (m, t, mode)
                y = mdl.NewBoolVar(""); ybin[key] = y; roles[m].append(y)
                for i in pool[t]:
                    if D[i]["size"] <= ub and (mode != "full" or not D[i]["isCopyrighted"]):
                        x[key][i] = mdl.NewBoolVar("")
                e = sum(D[i]["size"] * v for i, v in x[key].items())
                mdl.Add(e >= lb).OnlyEnforceIf(y)
                mdl.Add(e <= ub * y)
                energy.append(M[m]["energyCost"] * y)
                if mode == "src":
                    nsrc[t].append(y)
                elif mode == "full":
                    obj.append(M[m]["value"] * y)
                else:
                    obj.append((M[m]["value"] // 2) * y)
    # multi-type candidates: full (free pool items only) or half (any pool items), no sources
    multi_bins = {}
    for m in dict.fromkeys(multi_cands):
        if m in st.used:
            continue
        r = M[m]["dataRequirements"]
        if len(r) < 2 or any(q["upperBound"] < q["lowerBound"] for q in r.values()) or M[m]["energyCost"] > ES:
            continue
        for mode in ("full", "half"):
            ok = True
            for t, q in r.items():
                vol = sum(D[i]["size"] for i in pool[t] if mode == "half" or not D[i]["isCopyrighted"])
                if vol < q["lowerBound"]:
                    ok = False
            if not ok:
                continue
            y = mdl.NewBoolVar(""); roles[m].append(y)
            multi_bins[(m, mode)] = y
            for t, q in r.items():
                key = (m, t, "M" + mode)
                for i in pool[t]:
                    if D[i]["size"] <= q["upperBound"] and (mode == "half" or not D[i]["isCopyrighted"]):
                        x[key][i] = mdl.NewBoolVar("")
                e = sum(D[i]["size"] * v for i, v in x[key].items())
                mdl.Add(e >= q["lowerBound"]).OnlyEnforceIf(y)
                mdl.Add(e <= q["upperBound"] * y)
                ybin[key] = y
            energy.append(M[m]["energyCost"] * y)
            obj.append((M[m]["value"] if mode == "full" else M[m]["value"] // 2) * y)
    for m, vs in roles.items():
        if len(vs) > 1:
            mdl.AddAtMostOne(vs)
    byi = collections.defaultdict(list)
    for key, d in x.items():
        for i, v in d.items():
            byi[i].append(v)
    for i, vs in byi.items():
        mdl.AddAtMostOne(vs)
    for t in TYPES:
        mdl.Add(sum(ntgt[t]) <= sum(nsrc[t]))
    mdl.Add(sum(energy) <= ES)
    mdl.Maximize(sum(obj))
    if hint:
        on = set()
        for m, (kind, bins) in hint.items():
            nreq = len(M[m]["dataRequirements"])
            if kind == "tgt":
                if m in zt:
                    on.add(zt[m].Index())
                continue
            for t, its in bins.items():
                if nreq == 1:
                    key = (m, t, kind)
                else:
                    key = (m, t, "M" + kind)
                if key in ybin:
                    on.add(ybin[key].Index())
                    for i in its:
                        if i in x[key]:
                            on.add(x[key][i].Index())
        seen = set()
        for v in list(zt.values()) + list(ybin.values()) + [xv for d in x.values() for xv in d.values()]:
            if v.Index() in seen:
                continue
            seen.add(v.Index())
            mdl.AddHint(v, 1 if v.Index() in on else 0)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tlim
    sv.parameters.num_workers = workers
    r = sv.Solve(mdl)
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if log:
            print("resid: no solution", sv.StatusName(r))
        return 0
    gain = int(sv.ObjectiveValue())
    if log:
        print("resid gain", gain, sv.StatusName(r), "bound", sv.BestObjectiveBound(), "ES", ES, flush=True)
    if gain <= 0:
        return 0
    # apply
    new_src = collections.defaultdict(list)
    for key, y in ybin.items():
        if sv.Value(y):
            m, t, mode = key
            for i, v in x[key].items():
                if sv.Value(v):
                    st.item_bin[i] = (m, t); st.bin_items[(m, t)].add(i)
            st.used.add(m)
            if mode == "src":
                new_src[t].append(m)
    for m, v in zt.items():
        if sv.Value(v):
            t = next(iter(M[m]["dataRequirements"]))
            s = new_src[t].pop()
            st.src[s] = m; st.tgt[m].append(s); st.used.add(m)
    # unused new sources (if any) -> remove their data
    for t, lst in new_src.items():
        for s in lst:
            for i in list(st.bin_items[(s, t)]):
                del st.item_bin[i]
            st.bin_items[(s, t)] = set()
            st.used.discard(s)
    return gain


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    st = State(inst, sol)
    g = add_models(st, tlim=float(sys.argv[3]))
    sc = st.score()
    print(sc)
    if sc[1]:
        json.dump(st.to_sol(), open(sys.argv[4], "w"))
