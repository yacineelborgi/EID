"""Upgrade half models to full by freeing free datasets for their (small) requirements.
For each candidate half model (no source-covered requirement, all windows valid), for each of
its types: CP-SAT over {new free-only sink bin} + bins holding candidate free items + the
highest-waste free bins of that type; all old bins must stay in their windows.
If every requirement of the model gets free items, the model is upgraded (its copyrighted
items are released). Works on any instance."""
import json, sys, random, collections, time
from ortools.sat.python import cp_model
sys.path.insert(0, '.')
from sol6 import State


def free_bins(st, t, full):
    M = st.M
    out = {}
    for m in full:
        r = M[m]["dataRequirements"].get(t)
        if r:
            out[(m, t)] = (r["lowerBound"], r["upperBound"])
    return out


def try_sink(st, t, lb, ub, full, rng, ntop=40, ny=6, tlim=10.0):
    """Find a re-packing of some free bins of type t that frees free items summing into [lb, ub].
    Returns list of (item, newbin) moves + sink items, or None."""
    D = st.D
    fb = free_bins(st, t, full)
    waste = {k: sum(D[i]["size"] for i in st.bin_items[k]) - fb[k][0] for k in fb}
    # bins containing a free item <= ub
    ys = [k for k in fb if any(D[i]["size"] <= ub for i in st.bin_items[k])]
    rng.shuffle(ys)
    top = sorted(fb, key=lambda k: -waste[k])[:ntop]
    sel = list(dict.fromkeys(ys[:ny] + top))
    items = sorted(set(i for k in sel for i in st.bin_items[k]))
    pool = [d["id"] for d in D if d["dataType"] == t and not d["isCopyrighted"] and d["id"] not in st.item_bin]
    items += pool
    mdl = cp_model.CpModel()
    x = {}
    for bi, k in enumerate(sel):
        for i in items:
            if D[i]["size"] <= fb[k][1]:
                x[i, bi] = mdl.NewBoolVar("")
    sk = len(sel)
    for i in items:
        if D[i]["size"] <= ub:
            x[i, sk] = mdl.NewBoolVar("")
    byi = collections.defaultdict(list); byb = collections.defaultdict(list)
    for (i, b), v in x.items():
        byi[i].append(v); byb[b].append((i, v))
    for vs in byi.values():
        mdl.AddAtMostOne(vs)
    for bi, k in enumerate(sel):
        e = sum(D[i]["size"] * v for i, v in byb[bi])
        mdl.Add(e >= fb[k][0]); mdl.Add(e <= fb[k][1])
    e = sum(D[i]["size"] * v for i, v in byb[sk])
    mdl.Add(e >= lb); mdl.Add(e <= ub)
    for (i, b), v in x.items():
        mdl.AddHint(v, 1 if b < sk and st.item_bin.get(i) == sel[b] else 0)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tlim
    sv.parameters.num_workers = 1
    r = sv.Solve(mdl)
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    asg = {}
    for (i, b), v in x.items():
        if sv.Value(v):
            asg[i] = "SINK" if b == sk else sel[b]
    return sel, items, asg


def apply_moves(st, sel, items, asg, sink_key):
    for i in items:
        old = st.item_bin.get(i)
        if old is not None:
            st.bin_items[old].discard(i); del st.item_bin[i]
    for i, b in asg.items():
        k = sink_key if b == "SINK" else b
        st.item_bin[i] = k; st.bin_items[k].add(i)


def main():
    inst = json.load(open(sys.argv[1]))
    st = State(inst, json.load(open(sys.argv[2])))
    out = sys.argv[3]
    ncand = int(sys.argv[4]); tries = int(sys.argv[5]); seed = int(sys.argv[6])
    rng = random.Random(seed)
    M, D = st.M, st.D
    cov = st.covered()
    full = {m for m in st.used if m not in st.src and st.is_full(m)}
    half = [m for m in st.used if m not in st.src and m not in full]
    cands = []
    for m in half:
        r = M[m]["dataRequirements"]
        if any((m, t) in cov for t in r) or any(q["upperBound"] < q["lowerBound"] for q in r.values()):
            continue
        g = M[m]["value"] - M[m]["value"] // 2
        if g <= 0:
            continue
        tot = sum(q["lowerBound"] for q in r.values())
        cands.append((tot / g, m))
    cands.sort()
    base = st.score()[0]
    print("start", base, "candidates", len(cands), flush=True)
    gained = 0
    for _, m in cands[:ncand]:
        r = M[m]["dataRequirements"]
        snap = (dict(st.item_bin),)
        ok = True
        old_items = {t: list(st.bin_items[(m, t)]) for t in r}
        # release current (copyrighted) items of m
        for t in r:
            for i in old_items[t]:
                st.bin_items[(m, t)].discard(i); del st.item_bin[i]
        for t, q in r.items():
            done = False
            for _ in range(tries):
                res = try_sink(st, t, q["lowerBound"], q["upperBound"], full, rng)
                if res:
                    sel, items, asg = res
                    apply_moves(st, sel, items, asg, (m, t))
                    done = True
                    break
            if not done:
                ok = False
                break
        if ok:
            full.add(m)
            sc, valid, msg = st.score()
            if valid and sc > base:
                gained += sc - base
                base = sc
                json.dump(st.to_sol(), open(out, "w"))
                print(f"upgraded {m} -> {sc}", flush=True)
                continue
            ok = False
        # restore
        st.item_bin = snap[0]
        st.bin_items = collections.defaultdict(set)
        for d, b in st.item_bin.items():
            st.bin_items[b].add(d)
        full.discard(m)
        print(f"  fail {m}", flush=True)
    print("total gain", gained)


if __name__ == "__main__":
    main()
