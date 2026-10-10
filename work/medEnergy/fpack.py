"""F-only packer for one type of a 2_medium plan (full bins, free items). Reports waste; LNS until filled.
usage: fpack.py plans.json idx type tmax seed"""
import sys, json, time, random, os
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/medEnergy")
import medpack
from medpack import TypePack, initial2, lns_step, squeeze, M, D
def fstats(tp):
    w = sum(sum(tp.items[j][0] for j in tp.asg[b]) - tp.bins[b][1] for b in tp.asg)
    left = sum(tp.items[j][0] for j in tp.pool())
    return f"F failed {[tp.bins[b][1] for b in tp.failed]} waste {w} left {left}"
def make(plan, t, seed):
    tp = TypePack(plan, t, seed, lambda s: None)
    keep = [b for b in tp.bins if b[3]]; tp.bins = keep; tp.bidx = {b[0]: q for q, b in enumerate(keep)}
    tp.items = [x for x in tp.items if x[2]]
    return tp
def run(tp, tmax, log, theta=3000):
    initial2(tp, theta=theta)
    log(f"t{tp.t} initial {fstats(tp)}")
    t0 = time.time(); it = 0
    while tp.failed and time.time() - t0 < tmax:
        it += 1
        if it % 2 == 1:
            fr = squeeze(tp, nbins=tp.rng.choice([30, 40, 60, 80]), tl=15)
            if fr: log(f"  sq freed {fr} -> {fstats(tp)} {time.time()-t0:.0f}s")
        fl = sorted(tp.failed, key=lambda b: tp.bins[b][1])
        nf, dw = lns_step(tp, [tp.rng.choice(fl)], ndonor=tp.rng.choice([20, 30, 40, 60]), tl=10)
        if nf or dw: log(f"  lns filled {nf} freed {dw} -> {fstats(tp)} {time.time()-t0:.0f}s")
    log(f"t{tp.t} END {fstats(tp)} {time.time()-t0:.0f}s")
    return tp
if __name__ == "__main__":
    plans = json.load(open(sys.argv[1])); plans = [plans] if isinstance(plans, dict) else plans
    plan = plans[int(sys.argv[2])]; t = sys.argv[3]; tmax = float(sys.argv[4]); seed = int(sys.argv[5])
    log = lambda s: print(s, flush=True)
    tp = make(plan, t, seed); run(tp, tmax, log)
