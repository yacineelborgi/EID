"""Driver for sa_pack: re-pack one type of a 6_shortage-like solution with extra sink bins.
usage: python sa_drive.py inst sol type iters seed T0 T1 out sink_model...
Writes out the new solution (sinks are added as sources only if fully packed; the
caller wires them to targets)."""
import json, sys, subprocess, collections
sys.path.insert(0, '.')
from sol6 import State


def type_bins(st, t):
    M = st.M
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
    return bins


def run_sa(st, t, sinks, iters, seed, T0, T1, sink_w=1.0, exe="./sa_pack", K0=1.0, K1=500.0):
    """sinks: list of model ids (mono type t) to add as new A bins. Returns (deficit, pool, newbin)."""
    D = st.D
    bins = type_bins(st, t)
    for m in sinks:
        r = st.M[m]["dataRequirements"][t]
        bins.append(((m, t), r["lowerBound"], r["upperBound"], False))
    bidx = {b[0]: k + 1 for k, b in enumerate(bins)}
    items = [d["id"] for d in D if d["dataType"] == t]
    lines = [f"{len(items)} {len(bins) + 1} {iters} {seed} {T0} {T1} {K0} {K1}"]
    for i in items:
        b = st.item_bin.get(i)
        lines.append(f"{D[i]['size']} {0 if D[i]['isCopyrighted'] else 1} {bidx.get(b, 0)}")
    lines.append("0 0 0 0")
    sinkset = set((m, t) for m in sinks)
    for (k, lb, ub, fo) in bins:
        lines.append(f"{lb} {ub} {1 if fo else 0} {sink_w if k in sinkset else 1.0}")
    out = subprocess.run([exe], input="\n".join(lines) + "\n", capture_output=True, text=True).stdout.split("\n")
    asg = [int(x) for x in out[:len(items)]]
    last = out[len(items)].split()
    deficit, pool = float(last[1]), int(last[3])
    newbin = {}
    for i, b in zip(items, asg):
        newbin[i] = bins[b - 1][0] if b > 0 else None
    return deficit, pool, newbin, bins


def apply(st, t, newbin):
    for i, b in newbin.items():
        old = st.item_bin.get(i)
        if old is not None:
            st.bin_items[old].discard(i)
            del st.item_bin[i]
    for i, b in newbin.items():
        if b is not None:
            st.item_bin[i] = b
            st.bin_items[b].add(i)


if __name__ == "__main__":
    inst = json.load(open(sys.argv[1]))
    sol = json.load(open(sys.argv[2]))
    t = sys.argv[3]
    iters, seed, T0, T1 = int(float(sys.argv[4])), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])
    out = sys.argv[8]
    sinks = [int(x) for x in sys.argv[9:]]
    st = State(inst, sol)
    deficit, pool, newbin, bins = run_sa(st, t, sinks, iters, seed, T0, T1)
    print("deficit", deficit, "pool", pool, flush=True)
    sink_fill = {m: sum(st.D[i]["size"] for i, b in newbin.items() if b == (m, t)) for m in sinks}
    print("sink fills", sink_fill, [st.M[m]["dataRequirements"][t] for m in sinks])
    if deficit == 0:
        apply(st, t, newbin)
        json.dump({"dataMappings": [[d, b[0]] for d, b in sorted(st.item_bin.items())],
                   "modelMappings": [[s, tt] for s, tt in sorted(st.src.items())]}, open(out, "w"))
        print("wrote", out)
