"""plan -> per-type dpack initial (both orders, keep best) + endgame LNS -> assemble (+swap) -> save.
usage: dfix.py plan.json tmax_per_type out.json seed"""
import sys, json, time, random
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortB2")
import fixB
from fixB import TypePack, assemble, with_add, save_if_better, swap_drop, check, INST
from dpack import initial_d, endgame


def takeall(tp, log):
    for b in sorted(tp.failed, key=lambda b: -tp.bins[b][1]):
        k, L, U, F = tp.bins[b]
        pl = [j for j in tp.pool() if (tp.items[j][2] or not F)]
        s = sum(tp.items[j][0] for j in pl)
        if L <= s <= U:
            tp.set_bin(b, pl); tp.failed.discard(b)
            log(f"   t{tp.t} takeall filled bin L {L} with {s}")


def main():
    plan = json.load(open(sys.argv[1])); tmax = float(sys.argv[2]); out = sys.argv[3]; seed = int(sys.argv[4])
    log = lambda s: print(s, flush=True)
    tps = {}
    for t in "ntic":
        best = None
        for order in ("asc", "desc"):
            for sd in range(2):
                tp = TypePack(plan, t, seed * 10 + sd, log=lambda s: None)
                initial_d(tp, order=order, G0=20000 if sd == 0 else 15000, log=log)
                takeall(tp, log)
                st = tp.stats()
                key = (st["failed"], st["failL"] - st["left"])
                if best is None or key < best[0]:
                    best = (key, tp)
        tp = best[1]
        log(f"t{t} chosen {tp.stats()}")
        endgame(tp, tmax, log)
        takeall(tp, log)
        log(f"t{t} end {tp.stats()} failed {[tp.bins[b] for b in tp.failed]}")
        tps[t] = tp
    sol, E = assemble(plan, tps)
    sc = save_if_better(sol, log) if E <= INST["energyCap"] else -1
    log(f"assembled {sc} E-cap {E - INST['energyCap']}")
    sol2, added = with_add(plan, tps, seed)
    sc2 = save_if_better(sol2, log)
    json.dump(sol2, open(out, "w"))
    log(f"+add {sc2} added {added}")
    if any(tp.failed for tp in tps.values()):
        swap_drop(plan, tps, log)
        sol3, added = with_add(plan, tps, seed)
        sc3 = save_if_better(sol3, log)
        log(f"after swap +add {sc3}")
        if sc3 > sc2:
            json.dump(sol3, open(out, "w"))
    log("DONE")


if __name__ == "__main__":
    main()
