"""plan -> solve2 assignment restarts -> residual mip6 rounds. Saves best legit solution."""
import json, os, random, sys, time
sys.path.insert(0, '/home/user/EID')
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
os.chdir('/home/user/EID')
import solve2
import pack6
from solve import score_of, strict_ok
from enc import classify

PATH = '/home/user/EID/datasets/6_shortage.json'
TXT = open(PATH).read()
INST = json.loads(TXT)
M, D = INST['models'], INST['datasets']


def evaluate(sol):
    sc, ok, msg = score_of(TXT, sol)
    return sc if (ok and strict_ok(INST, sol)) else -1


def int_energy(sol):
    dm, mm, full, half, us = classify(INST, sol)
    return sum(M[i]['energyCost'] for i in full | half | us)


def build6(plan, rnd, noise, order='lb_asc', exclude=(), base=None, use_rescue=True, log=print):
    import lnsfix
    dm, failed, _, _ = solve2.assign2(INST, plan, rnd, noise, exclude=exclude, order=order)
    nf0 = len(failed)
    for _ in range(4):
        nb = len(failed)
        failed = solve2.repair(INST, plan, dm, failed)
        if len(failed) == nb:
            break
    nf1 = len(failed)
    if use_rescue and failed:
        failed = lnsfix.rescue(INST, plan, dm, failed, rnd, log=None)
    log(f"    fails: {nf0} -> repair {nf1} -> rescue {len(failed)}")
    dm, mm, full, half, us, spare = solve2.finalize2(INST, plan, dm, failed)
    if base is not None:
        bdm, bmm, bfull, bhalf, bus = base
        dm.update(bdm); mm = bmm + mm
        full |= bfull; half |= bhalf; us |= bus
    pr = [i for i in plan["half"] + plan["full"] if i not in full and i not in half]
    added = solve2.greedy_add2(INST, dm, mm, full, half, us, spare, rnd, noise, priority=pr)
    sol = {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": mm}
    return sol, len(failed), added


def restarts(plan, n, seed0=0, log=print):
    best = (-1, None)
    t0 = time.time()
    for k in range(n):
        rnd = random.Random(seed0 + k)
        noise = 0.0 if k == 0 else 0.02 * (1 + k % 5)
        sol, nf, added = build6(plan, rnd, noise, 'lb_asc', log=log)
        sc = evaluate(sol)
        log(f" restart {k} noise {noise:.2f}: failed {nf} added {added} score {sc} t={time.time()-t0:.0f}s")
        if sc > best[0]:
            best = (sc, sol)
    return best


def resid_round(sol, tl=60, log=print):
    import mip6
    import numpy as np
    dm, mm, full, half, us = classify(INST, sol)
    used = full | half | us
    energy = sum(M[i]['energyCost'] for i in used)
    items = {t: ([], []) for t in 'ntic'}
    for d in D:
        if d['id'] not in dm:
            items[d['dataType']][d['isCopyrighted']].append(d['size'])
    left_e = INST['energyCap'] - energy
    tot = {t: sum(items[t][0]) + sum(items[t][1]) for t in 'ntic'}
    off = set(used)
    nsrc = {t: 0 for t in 'ntic'}
    for i, m in enumerate(M):
        if i in off: continue
        if m['energyCost'] > left_e: off.add(i); continue
    log(f" resid: left energy {left_e} items {[len(items[t][0])+len(items[t][1]) for t in 'ntic']} tot {tot} cand {len(M)-len(off)}")
    mo = mip6.build(INST, cap=left_e, forced_off=sorted(off), resid=items, jmax=200)
    h = mip6.to_highs(mo, integer=True)
    h.setOptionValue('time_limit', tl)
    h.run()
    x = np.array(h.getSolution().col_value)
    if len(x) == 0:
        return None
    rp = mip6.plan_from_x(mo, x)
    rp['obj'] = h.getInfo().objective_function_value
    log(f"  resid plan obj {rp['obj']} full {len(rp['full'])} half {len(rp['half'])} src {len(rp['source'])}")
    best = None
    for k in range(6):
        rnd = random.Random(1000 + k)
        s2, nf, added = solve2.build_solution(INST, rp, rnd, 0.0 if k == 0 else 0.05, 'lb_asc' if k % 2 == 0 else 'lb_desc',
                                              exclude=dm.keys(), base=(dict(dm), list(mm), set(full), set(half), set(us)))
        sc = evaluate(s2)
        log(f"   try {k}: failed {nf} added {added} score {sc}")
        if sc > (best[0] if best else -1):
            best = (sc, s2)
    return best


if __name__ == '__main__':
    plan = json.load(open(sys.argv[1]))
    n = int(sys.argv[2]); rounds = int(sys.argv[3]); out = sys.argv[4]
    seed0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    log = lambda s: print(s, flush=True)
    log(f"plan obj {plan.get('obj')} full {len(plan['full'])} half {len(plan['half'])} src {len(plan['source'])}")
    sc, sol = restarts(plan, n, seed0, log)
    log(f"best after restarts {sc}")
    json.dump(sol, open(out, 'w'))
    for r in range(rounds):
        b = resid_round(sol, log=log)
        if b is None or b[0] <= sc:
            break
        sc, sol = b
        json.dump(sol, open(out, 'w'))
        log(f" round {r}: {sc}")
    log(f"FINAL {sc} energy {int_energy(sol)} cap {INST['energyCap']}")
