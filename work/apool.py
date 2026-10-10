"""Per-type A-pool re-optimization for 6_shortage with CP-SAT.
Keeps full models (free pool) fixed. Re-chooses which mono models serve as sources of
type t and how the A-available items (copyrighted + free items not used by full models)
are packed into those sources and into the fixed half-model bins.
Objective: max number of sources (coverages), energy-limited."""
import json, sys, time, bisect, collections
from ortools.sat.python import cp_model
sys.path.insert(0, '.')
from sol6 import State


def build(st, t, emax=1e11, extra_energy=0, single_lb=0, tlim=60, workers=4, hint=True, log=False,
          min_extra=0):
    M, D = st.M, st.D
    full = {m for m in st.used if m not in st.src and st.is_full(m)}
    cov = st.covered()
    fixed = []
    for m in st.used:
        if m in st.src or m in full:
            continue
        r = M[m]["dataRequirements"].get(t)
        if r and (m, t) not in cov:
            fixed.append(((m, t), r["lowerBound"], r["upperBound"]))
    cur_src = [s for s in st.src if next(iter(M[s]["dataRequirements"])) == t]
    N = len(cur_src)
    E_cur = sum(M[s]["energyCost"] for s in cur_src)
    usedF = set(i for m in full for i in st.bin_items.get((m, t), ()))
    items = [d["id"] for d in D if d["dataType"] == t and d["id"] not in usedF]
    size = {i: D[i]["size"] for i in items}
    cand = []
    for i, m in enumerate(M):
        r = m["dataRequirements"]
        if len(r) != 1 or t not in r:
            continue
        if i in st.used and i not in st.src:
            continue
        lb, ub = r[t]["lowerBound"], r[t]["upperBound"]
        if ub < lb:
            continue
        if m["energyCost"] > emax and i not in st.src:
            continue
        cand.append((i, lb, ub, m["energyCost"]))
    mdl = cp_model.CpModel()
    z = {}
    x = collections.defaultdict(dict)
    srt = sorted(items, key=lambda i: size[i])
    svals = [size[i] for i in srt]
    for (s, lb, ub, E) in cand:
        z[s] = mdl.NewBoolVar("")
        if lb <= single_lb:
            lo = bisect.bisect_left(svals, lb); hi = bisect.bisect_right(svals, ub)
            its = srt[lo:hi]
        else:
            hi = bisect.bisect_right(svals, ub)
            its = srt[:hi]
        for i in its:
            x[("s", s)][i] = mdl.NewBoolVar("")
        e = sum(size[i] * v for i, v in x[("s", s)].items())
        mdl.Add(e >= lb).OnlyEnforceIf(z[s])
        mdl.Add(e <= ub * z[s])
        if lb <= single_lb:
            mdl.Add(sum(x[("s", s)].values()) <= z[s])
    for (k, lb, ub) in fixed:
        hi = bisect.bisect_right(svals, ub)
        for i in srt[:hi]:
            x[("b", k)][i] = mdl.NewBoolVar("")
        e = sum(size[i] * v for i, v in x[("b", k)].items())
        mdl.Add(e >= lb); mdl.Add(e <= ub)
    byi = collections.defaultdict(list)
    for b, d in x.items():
        for i, v in d.items():
            byi[i].append(v)
    for i, vs in byi.items():
        mdl.AddAtMostOne(vs)
    ncov = sum(z.values())
    energy = sum(int(E) * z[s] for (s, lb, ub, E) in cand)
    mdl.Add(energy <= int(E_cur + extra_energy))
    mdl.Add(ncov >= N + min_extra)
    mdl.Maximize(ncov)
    if hint:
        cs = set(cur_src)
        for (s, lb, ub, E) in cand:
            mdl.AddHint(z[s], 1 if s in cs else 0)
        for b, d in x.items():
            key = (b[1], t) if b[0] == "s" else b[1]
            for i, v in d.items():
                mdl.AddHint(v, 1 if st.item_bin.get(i) == key else 0)
    nvars = sum(len(d) for d in x.values())
    if log:
        print(f"type {t}: items {len(items)} fixed bins {len(fixed)} cand sources {len(cand)} cur N {N} vars {nvars}", flush=True)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tlim
    sv.parameters.num_workers = workers
    r = sv.Solve(mdl)
    res = None
    if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sel = [s for s in z if sv.Value(z[s])]
        asg = {}
        for b, d in x.items():
            for i, v in d.items():
                if sv.Value(v):
                    asg[i] = b
        res = dict(sources=sel, asg=asg, ncov=len(sel), N=N, energy=sv.Value(energy), E_cur=E_cur,
                   status=sv.StatusName(r), bound=sv.BestObjectiveBound())
    elif log:
        print("status", sv.StatusName(r))
    return res


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    t = sys.argv[3]
    st = State(inst, sol)
    res = build(st, t, emax=float(sys.argv[4]), extra_energy=float(sys.argv[5]), tlim=float(sys.argv[6]), log=True,
                min_extra=int(sys.argv[7]) if len(sys.argv) > 7 else 0)
    if res:
        print({k: v for k, v in res.items() if k not in ("asg", "sources")})
