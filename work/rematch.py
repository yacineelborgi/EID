"""Re-match closers globally: for each multi-item bin keep all items but one as base,
then solve one min-cost assignment of closers (and single bins) over all non-base items."""
import json, sys, collections
sys.path.insert(0, '.')
from pack6b import closer_match
from sol6 import State


def rematch_type(st, t, full, pick="smallest"):
    M, D = st.M, st.D
    cov = st.covered()
    bins = []
    for m in st.used:
        r = M[m]["dataRequirements"].get(t)
        if not r or ((m, t) in cov and m not in st.src):
            continue
        bins.append(((m, t), r["lowerBound"], r["upperBound"], m in full))
    base = {}
    reqs = {True: [], False: []}
    for k, lb, ub, fo in bins:
        its = sorted(st.bin_items[k], key=lambda i: D[i]["size"])
        if len(its) >= 2:
            c = its[0] if pick == "smallest" else its[-1]
            b = [i for i in its if i != c]
            bs = sum(D[i]["size"] for i in b)
            base[k] = b
            reqs[fo].append((k, lb - bs, ub - lb))
        else:
            base[k] = []
            reqs[fo].append((k, lb, ub - lb))
    inbase = set(i for b in base.values() for i in b)
    free = [(d["size"], d["id"], True) for d in D if d["dataType"] == t and not d["isCopyrighted"] and d["id"] not in inbase]
    rF, uF = closer_match(reqs[True], free)
    usedF = set(i[1] for i in rF.values())
    allit = [(d["size"], d["id"], not d["isCopyrighted"]) for d in D if d["dataType"] == t and d["id"] not in inbase and d["id"] not in usedF]
    rA, uA = closer_match(reqs[False], allit)
    res = dict(rF); res.update(rA)
    return base, res, uF + uA


if __name__ == "__main__":
    inst = json.load(open("../datasets/6_shortage.json"))
    sol = json.load(open(sys.argv[1]))
    st = State(inst, sol)
    full = {m for m in st.used if m not in st.src and st.is_full(m)}
    D = st.D
    for pick in ("smallest", "largest"):
        tot = 0
        for t in "ntic":
            base, res, un = rematch_type(st, t, full, pick)
            used = set(i for b in base.values() for i in b) | set(it[1] for it in res.values())
            left = sum(d["size"] for d in D if d["dataType"] == t and d["id"] not in used)
            print(pick, t, "unmatched", len(un), "left volume", left)
            tot += left
        print(pick, "total left", tot)
