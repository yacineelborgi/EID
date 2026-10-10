"""usage: sarun.py plan.json sol.json secs_per_type seed T0 T1 eps out.json [types]"""
import sys, json, subprocess, os, random
D0 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D0); sys.path.insert(0, "/home/user/EID")
from ptype import type_data, M, D, INST
import fix

class TP:
    def __init__(s, t, bins, items, asg):
        s.t, s.bins, s.items = t, bins, items
        s.asg = asg

def sol_assign(plan, sol, t, bins, items):
    dm = {d: m for d, m in sol["dataMappings"]}
    bidx = {b[0]: q for q, b in enumerate(bins)}
    jidx = {it[1]: j for j, it in enumerate(items)}
    a = [-1] * len(items)
    for d, m in dm.items():
        if D[d]["dataType"] == t and m in bidx:
            a[jidx[d]] = bidx[m]
    return a

PEN = float(os.environ.get("PEN", "300"))
def run_sa(t, bins, items, a, secs, seed, T0, T1, eps, W=None, tag=""):
    fn = f"{D0}/work_{t}{tag}.in"; fo = f"{D0}/work_{t}{tag}.out"
    with open(fn, "w") as f:
        f.write(f"{len(bins)} {len(items)}\n")
        for q, (k, L, U, isF) in enumerate(bins):
            f.write(f"{L} {U} {int(isF)} {W[q] if W else 1.0}\n")
        for s, _, fr in items:
            f.write(f"{s} {int(fr)}\n")
        for x in a:
            f.write(f"{x}\n")
    r = subprocess.run([f"{D0}/sa", fn, fo, str(secs), str(seed), str(T0), str(T1), str(eps), str(PEN)], capture_output=True, text=True)
    print(f"  t{t}: " + r.stderr.strip().replace("\n", " | "), flush=True)
    return [int(x) for x in open(fo).read().split()]

def to_tp(t, bins, items, a):
    S = [0] * len(bins); per = {}
    for j, b in enumerate(a):
        if b >= 0:
            S[b] += items[j][0]; per.setdefault(b, []).append(j)
    asg = {b: js for b, js in per.items() if bins[b][1] <= S[b] <= bins[b][2]}
    return TP(t, bins, items, asg)

def main():
    plan = json.load(open(sys.argv[1])); sol = json.load(open(sys.argv[2]))
    secs, seed, T0, T1, eps = float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    out = sys.argv[8]; types = sys.argv[9] if len(sys.argv) > 9 else "ntic"
    tps = {}
    for t in "ntic":
        bins, items = type_data(plan, t)
        a = sol_assign(plan, sol, t, bins, items)
        if t in types:
            a = run_sa(t, bins, items, a, secs, seed, T0, T1, eps)
        tps[t] = to_tp(t, bins, items, a)
        fl = [bins[b] for b in range(len(bins)) if b not in tps[t].asg]
        print(f"t{t} failed {len(fl)} {fl[:6]}", flush=True)
    s1, E = fix.assemble(plan, tps)
    sc1 = fix.save_if_better(s1)
    s2, added = fix.with_add(plan, tps, seed)
    sc2 = fix.save_if_better(s2)
    print("score", sc1, "with_add", sc2, added, flush=True)
    json.dump(s2 if sc2 > sc1 else s1, open(out, "w"))

if __name__ == "__main__":
    main()
