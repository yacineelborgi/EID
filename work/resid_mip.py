"""HiGHS version of resid_add (run as a separate process: highspy and ortools clash).
usage: python resid_mip.py inst sol_in params.json sol_out
params: {"multi_cands": [...], "extra_mono": [...], "hint": {model: [kind, {t: [items]}]},
         "tlim": 60, "cap_src": 120, "cap_data": 120, "emax_src": 3e11, "max_tgt": 300}
Adds models using pool datasets and spare energy (exact packing), maximizing value."""
import json, sys, collections
import numpy as np
import highspy
from scipy.sparse import coo_matrix
sys.path.insert(0, '.')
from sol6 import State, TYPES


def solve(st, multi_cands=(), extra_mono=(), hint=None, tlim=60, cap_src=120, cap_data=120, emax_src=3e11, max_tgt=300,
          threads=1, log=False, refill=(), refill_hint=None, max_mtgt=200, vol=False):
    M, D = st.M, st.D
    cap = st.inst["energyCap"]
    for m, t, fo in refill:  # kept models whose bins were emptied are still trained
        st.used.add(m)
    ES = cap - sum(M[m]["energyCost"] for m in st.used)
    pool = {t: [d["id"] for d in D if d["dataType"] == t and d["id"] not in st.item_bin] for t in TYPES}
    unused = [i for i in range(len(M)) if i not in st.used]
    hint = hint or {}
    forced = set(extra_mono) | set(int(k) for k in hint)
    cost, lo, hi = [], [], []
    rows, cols, vals, rlo, rhi = [], [], [], [], []
    nr = [0]

    def var(c):
        cost.append(float(c)); return len(cost) - 1

    def row(ent, a, b):
        for c, v in ent:
            rows.append(nr[0]); cols.append(c); vals.append(v)
        rlo.append(a); rhi.append(b); nr[0] += 1

    zt = {}
    ybin = {}      # key (m, t, mode) -> y col
    xv = collections.defaultdict(dict)  # key -> item -> col
    roles = collections.defaultdict(list)
    nsrc = collections.defaultdict(list); ntgt = collections.defaultdict(list)
    energy = []
    LAM, MU = 2.8e-10, 0.0052
    for t in ([] if vol else TYPES):
        pv = sum(D[i]["size"] for i in pool[t])
        pf = sum(D[i]["size"] for i in pool[t] if not D[i]["isCopyrighted"])
        monos = [i for i in unused if len(M[i]["dataRequirements"]) == 1 and t in M[i]["dataRequirements"]]
        tg = sorted(monos, key=lambda i: -(M[i]["value"] // 2) / (M[i]["energyCost"] + 1))[:max_tgt]
        tg = list(dict.fromkeys(tg + [i for i in monos if i in forced]))
        for m in tg:
            if M[m]["value"] // 2 <= 0 or M[m]["energyCost"] > ES:
                continue
            c = var(M[m]["value"] // 2); zt[m] = c; roles[m].append(c); ntgt[t].append(c)
            energy.append((c, M[m]["energyCost"]))
        lbof = lambda m: M[m]["dataRequirements"][t]["lowerBound"]
        okm = [m for m in monos if M[m]["dataRequirements"][t]["upperBound"] >= lbof(m) and lbof(m) <= pv and M[m]["energyCost"] <= ES]
        top_src = set(sorted((m for m in okm if M[m]["energyCost"] <= emax_src),
                             key=lambda m: LAM * M[m]["energyCost"] + MU * lbof(m))[:cap_src])
        top_full = set(sorted((m for m in okm if lbof(m) <= pf),
                              key=lambda m: -(M[m]["value"] - LAM * M[m]["energyCost"] - MU * lbof(m)))[:cap_data])
        top_half = set(sorted(okm, key=lambda m: -(M[m]["value"] // 2 - LAM * M[m]["energyCost"] - MU * lbof(m)))[:cap_data // 2])
        for m in okm:
            lb, ub = lbof(m), M[m]["dataRequirements"][t]["upperBound"]
            modes = []
            if (M[m]["energyCost"] <= emax_src or m in forced) and (m in top_src or m in forced):
                modes.append("src")
            if lb <= pf and (m in top_full or m in forced):
                modes.append("full")
            if m in top_half or m in forced:
                modes.append("half")
            for mode in modes:
                val = 0 if mode == "src" else (M[m]["value"] if mode == "full" else M[m]["value"] // 2)
                y = var(val); key = (m, t, mode); ybin[key] = y; roles[m].append(y)
                energy.append((y, M[m]["energyCost"]))
                if mode == "src":
                    nsrc[t].append(y)
                for i in pool[t]:
                    if D[i]["size"] <= ub and (mode != "full" or not D[i]["isCopyrighted"]):
                        xv[key][i] = var(0)
    # multi-type targets fully covered by new sources (value v//2)
    mt = [] if vol else [i for i in unused if len(M[i]["dataRequirements"]) > 1 and M[i]["value"] // 2 > 0 and M[i]["energyCost"] <= ES]
    mt.sort(key=lambda i: -(M[i]["value"] // 2) / (LAM * M[i]["energyCost"] + 125 * len(M[i]["dataRequirements"])))
    mt = list(dict.fromkeys(mt[:max_mtgt] + [i for i in mt if i in forced or str(i) in hint or i in hint]))
    for m in mt:
        c = var(M[m]["value"] // 2); zt[m] = c; roles[m].append(c)
        energy.append((c, M[m]["energyCost"]))
        for t in M[m]["dataRequirements"]:
            ntgt[t].append(c)
    # multi-type candidates
    for m in ([] if vol else dict.fromkeys(int(k) for k in multi_cands)):
        if m in st.used:
            continue
        r = M[m]["dataRequirements"]
        if len(r) < 2 or any(q["upperBound"] < q["lowerBound"] for q in r.values()) or M[m]["energyCost"] > ES:
            continue
        for mode in ("full", "half"):
            if any(sum(D[i]["size"] for i in pool[t] if mode == "half" or not D[i]["isCopyrighted"]) < q["lowerBound"] for t, q in r.items()):
                continue
            y = var(M[m]["value"] if mode == "full" else M[m]["value"] // 2)
            roles[m].append(y); energy.append((y, M[m]["energyCost"]))
            for t, q in r.items():
                key = (m, t, "M" + mode); ybin[key] = y
                for i in pool[t]:
                    if D[i]["size"] <= q["upperBound"] and (mode == "half" or not D[i]["isCopyrighted"]):
                        xv[key][i] = var(0)
    # bins of kept models whose items were released: must be refilled (no y variable)
    rkeys = []
    for m, t, fo in refill:
        key = (m, t, "R")
        rkeys.append(key)
        q = M[m]["dataRequirements"][t]
        for i in pool[t]:
            if D[i]["size"] <= q["upperBound"] and (not fo or not D[i]["isCopyrighted"]):
                if fo is False and not D[i]["isCopyrighted"] and vol:
                    xv[key][i] = var(-D[i]["size"] * 1.001)  # prefer keeping free datasets in the pool
                else:
                    xv[key][i] = var(-D[i]["size"] if vol else 0)
    # bin constraints
    byi = collections.defaultdict(list)
    for key in rkeys:
        m, t, _ = key
        q = M[m]["dataRequirements"][t]
        d = xv[key]
        row([(c, D[i]["size"]) for i, c in d.items()], q["lowerBound"], q["upperBound"])
        for i, c in d.items():
            byi[i].append(c)
    for key, d in xv.items():
        m, t, mode = key
        if mode == "R":
            continue
        q = M[m]["dataRequirements"][t]
        y = ybin[key]
        row([(c, D[i]["size"]) for i, c in d.items()] + [(y, -q["lowerBound"])], 0, np.inf)
        row([(c, D[i]["size"]) for i, c in d.items()] + [(y, -q["upperBound"])], -np.inf, 0)
        for i, c in d.items():
            row([(c, 1), (y, -1)], -np.inf, 0)
            byi[i].append(c)
    for key, y in ybin.items():
        if key not in xv:  # no candidate items at all
            row([(y, 1)], -np.inf, 0)
    for i, cs in byi.items():
        if len(cs) > 1:
            row([(c, 1) for c in cs], -np.inf, 1)
    for m, cs in roles.items():
        cs = list(dict.fromkeys(cs))
        if len(cs) > 1:
            row([(c, 1) for c in cs], -np.inf, 1)
    for t in TYPES:
        if ntgt[t]:
            row([(c, 1) for c in ntgt[t]] + [(c, -1) for c in nsrc[t]], -np.inf, 0)
    seen = set(); ent = []
    for c, e in energy:
        if c not in seen:
            seen.add(c); ent.append((c, e / 1e9))
    row(ent, -np.inf, ES / 1e9)
    n = len(cost)
    A = coo_matrix((vals, (rows, cols)), shape=(nr[0], n)).tocsc()
    x0 = None
    if hint or refill:
        x0 = np.zeros(n)
        for mk, (kind, bins) in hint.items():
            m = int(mk)
            if kind == "tgt":
                if m in zt:
                    x0[zt[m]] = 1
                continue
            for t, its in bins.items():
                key = (m, t, kind) if len(M[m]["dataRequirements"]) == 1 else (m, t, "M" + kind)
                if key in ybin:
                    x0[ybin[key]] = 1
                    for i in its:
                        if i in xv[key]:
                            x0[xv[key][i]] = 1
        for (m, t, fo), its in zip(refill, refill_hint or [[] for _ in refill]):
            key = (m, t, "R")
            for i in its:
                if i in xv[key]:
                    x0[xv[key][i]] = 1
    import solver
    sol, gain, bound = solver.milp(np.array(cost), A, rlo, rhi, "max", tlim, threads, x0, 0.0, log)
    if not np.isfinite(gain):
        # no solution found: keep the hint (it is feasible by construction)
        sol, gain = (x0 if x0 is not None else np.zeros(n)), float(np.dot(cost, x0 if x0 is not None else np.zeros(n)))

    class _I:
        pass
    info = _I(); info.objective_function_value = gain; info.mip_dual_bound = bound
    # apply
    for key in rkeys:
        m, t, _ = key
        for i, c in xv[key].items():
            if sol[c] > 0.5:
                st.item_bin[i] = (m, t); st.bin_items[(m, t)].add(i)
    new_src = collections.defaultdict(list)
    done = set()
    for key, y in ybin.items():
        if sol[y] > 0.5:
            m, t, mode = key
            for i, c in xv[key].items():
                if sol[c] > 0.5:
                    st.item_bin[i] = (m, t); st.bin_items[(m, t)].add(i)
            st.used.add(m)
            if mode == "src" and m not in done:
                new_src[t].append(m); done.add(m)
    for m, c in zt.items():
        if sol[c] > 0.5:
            for t in M[m]["dataRequirements"]:
                s = new_src[t].pop()
                st.src[s] = m; st.tgt[m].append(s)
            st.used.add(m)
    for t, lst in new_src.items():
        for s in lst:
            for i in list(st.bin_items[(s, t)]):
                del st.item_bin[i]
            st.bin_items[(s, t)] = set()
            st.used.discard(s)
    return gain, info.mip_dual_bound


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    st = State(inst, json.load(open(sys.argv[2])))
    p = json.load(open(sys.argv[3]))
    hint = {int(k): (v[0], v[1]) for k, v in p.get("hint", {}).items()}
    g, b = solve(st, p.get("multi_cands", []), p.get("extra_mono", []), hint, p.get("tlim", 60),
                 p.get("cap_src", 120), p.get("cap_data", 120), p.get("emax_src", 3e11), p.get("max_tgt", 300),
                 p.get("threads", 1), refill=[tuple(r) for r in p.get("refill", [])], refill_hint=p.get("refill_hint"),
                 vol=p.get("vol", False))
    print("gain", g, "bound", b, flush=True)
    json.dump(st.to_sol(), open(sys.argv[4], "w"))
