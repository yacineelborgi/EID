"""fpack v2: F-only packer with waste-targeted squeezes, state save/restore, final DP fill of failed bins.
usage: fpack2.py plans.json idx type tmax seed [state_in] state_out"""
import sys, json, time, random, os
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/medEnergy")
from fpack import make, fstats
from medpack import initial2, lns_step, mip_assign
sys.path.insert(0, "/home/user/EID")
from exact_pack import dp_min_subset
def wst(tp, b): return sum(tp.items[j][0] for j in tp.asg[b]) - tp.bins[b][1]
def diag(tp, log):
    import collections
    c = collections.Counter(); n = collections.Counter()
    for b in tp.asg:
        L = tp.bins[b][1]; k = 0 if L <= 1 else 1 if L <= 3000 else 2 if L <= 10000 else 3 if L <= 20000 else 4
        c[k] += wst(tp, b); n[k] += (wst(tp, b) > 0)
    log(f"   waste by class (<=1,<=3k,<=10k,<=20k,>20k): {dict(c)} nonzero {dict(n)}")
def sq2(tp, bl, tl):
    bins, items = tp.bins, tp.items
    pl = tp.pool(); its = pl + [j for b in bl for j in tp.asg[b]]
    pos = {j: q for q, j in enumerate(its)}
    start = {q: [pos[j] for j in tp.asg[b]] for q, b in enumerate(bl)}
    cur = sum(items[j][0] for b in bl for j in tp.asg[b])
    r = mip_assign([bins[b] for b in bl], [items[j] for j in its], tl=tl, start=start)
    if r is None: return 0
    st, out, filled_, objv, bnd = r
    new = sum(items[its[j]][0] for q in range(len(bl)) for j in out[q])
    if new >= cur: return 0
    for b in bl: tp.set_bin(b, [])
    for q, b in enumerate(bl): tp.set_bin(b, [its[j] for j in out[q]])
    return cur - new
def try_fill(tp, log):
    for b in sorted(tp.failed, key=lambda b: -tp.bins[b][1]):
        pl = tp.pool(); its = [(tp.items[j][0], j) for j in pl]
        r = dp_min_subset(its, tp.bins[b][1], tp.bins[b][2])
        if r is not None:
            tp.set_bin(b, [j for _, j in r]); tp.failed.discard(b); log(f"   DP filled bin lb {tp.bins[b][1]}")
if __name__ == "__main__":
    plans = json.load(open(sys.argv[1])); plans = [plans] if isinstance(plans, dict) else plans
    plan = plans[int(sys.argv[2])]; t = sys.argv[3]; tmax = float(sys.argv[4]); seed = int(sys.argv[5])
    sin = sys.argv[6] if len(sys.argv) > 7 else None; sout = sys.argv[-1]
    log = lambda s: print(s, flush=True)
    tp = make(plan, t, seed)
    if sin and os.path.exists(sin):
        S = json.load(open(sin)); idx = {it[1]: j for j, it in enumerate(tp.items)}
        for k, ids in S.items():
            b = tp.bidx[int(k)]; tp.set_bin(b, [idx[d] for d in ids])
        tp.failed = {b for b in range(len(tp.bins)) if b not in tp.asg}
    else:
        initial2(tp)
    log(f"t{t} start {fstats(tp)}"); diag(tp, log)
    t0 = time.time(); it = 0
    def save():
        json.dump({str(tp.bins[b][0]): [tp.items[j][1] for j in js] for b, js in tp.asg.items()}, open(sout, "w"))
    while tp.failed and time.time() - t0 < tmax:
        it += 1
        try_fill(tp, log)
        if not tp.failed: break
        mode = it % 3
        filled = list(tp.asg)
        pos = [b for b in filled if wst(tp, b) > 0 and tp.bins[b][1] > 1]
        if mode == 0:
            bl = tp.rng.sample(pos, min(len(pos), 50))
        elif mode == 1:
            bl = sorted(pos, key=lambda b: -wst(tp, b) * (0.5 + tp.rng.random()))[:40] + tp.rng.sample(filled, 25)
        else:
            L0 = tp.rng.choice([tp.bins[b][1] for b in pos]) if pos else 5000
            bl = sorted(filled, key=lambda b: abs(tp.bins[b][1] - L0) * (0.5 + tp.rng.random()))[:60]
        bl = list(dict.fromkeys(bl))
        fr = sq2(tp, bl, tl=20)
        if fr: log(f"  it {it} m{mode} sq freed {fr} -> {fstats(tp)} {time.time()-t0:.0f}s"); save()
    try_fill(tp, log)
    save(); diag(tp, log)
    log(f"t{t} END {fstats(tp)} {time.time()-t0:.0f}s")
