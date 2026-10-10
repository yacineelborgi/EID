"""per-type pack: trun.py plan.json sol.json types seed tmax outprefix [init=sol|d]
saves outprefix_<t>.json with model -> item ids, failed models, stats"""
import sys, json, time, os
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortB2")
import fixB
from fixB import TypePack, packs_from_sol, rematch_small, squeeze, lns_step, D


def dump(tp, path):
    asg = {str(tp.bins[b][0]): [tp.items[j][1] for j in js] for b, js in tp.asg.items()}
    st = tp.stats()
    json.dump({"t": tp.t, "asg": asg, "failed": [tp.bins[b][0] for b in tp.failed], "stats": st}, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)


def takeall(tp, log):
    for b in sorted(tp.failed, key=lambda b: -tp.bins[b][1]):
        k, L, U, F = tp.bins[b]
        pl = [j for j in tp.pool() if (tp.items[j][2] or not F)]
        s = sum(tp.items[j][0] for j in pl)
        if L <= s <= U:
            tp.set_bin(b, pl); tp.failed.discard(b); log(f"  t{tp.t} takeall L {L} with {s}")


def main():
    plan = json.load(open(sys.argv[1])); ts = sys.argv[3]; seed = int(sys.argv[4]); tmax = float(sys.argv[5])
    outp = sys.argv[6]; init = sys.argv[7] if len(sys.argv) > 7 else "sol"
    log = lambda s: print(s, flush=True)
    if init == "sol":
        sol0 = json.load(open(sys.argv[2]))
        tps = packs_from_sol(plan, sol0, seed)
    for t in ts:
        if init == "sol":
            tp = tps[t]
        else:
            from dpack import initial_d
            tp = TypePack(plan, t, seed, log=lambda s: None)
            initial_d(tp, order="asc", log=log)
        log(f"t{t} start {tp.stats()}")
        if tp.failed and init == "sol":
            rematch_small(tp, log=log)
        t0 = time.time(); it = 0
        path = f"{outp}_{t}.json"
        best = None
        while tp.failed and time.time() - t0 < tmax:
            it += 1
            if it % 2 == 1:
                fr = squeeze(tp, nbins=tp.rng.choice([30, 40, 60]), tl=15)
                if fr:
                    log(f"  t{t} squeeze freed {fr} -> {tp.stats()}")
            fl = sorted(tp.failed, key=lambda b: tp.bins[b][1])
            k = min(len(fl), tp.rng.choice([1, 1, 2]))
            b0s = tp.rng.sample(fl, k)
            nd = tp.rng.choice([20, 30, 40, 60])
            nf, dw = lns_step(tp, b0s, ndonor=nd, tl=10)
            if nf or dw:
                log(f"  t{t} it {it} t={time.time()-t0:.0f}s nd {nd} filled {nf} freed {dw} -> {tp.stats()}")
            takeall(tp, log)
        takeall(tp, log)
        log(f"t{t} end {tp.stats()} failed {[tp.bins[b] for b in tp.failed]}")
        dump(tp, path)


if __name__ == "__main__":
    main()
