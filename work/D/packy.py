"""Pack a plan with seeded small fills. usage: packy.py plans.json idx tmax_per_type seed out.json"""
import sys, json, time, os, subprocess
SP='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/'
sys.path.insert(0, SP+'medEnergy')
import medpack
from medpack import TypePack, lns_step, squeeze, M, D, INST, assemble, check
from ptype import match_small, Packer, SortedList
from solve import greedy_add
def build_tp(plan, t, seed):
    tp = TypePack(plan, t, seed, lambda s: None)
    idx = {it[1]: j for j, it in enumerate(tp.items)}
    for k, dids in plan.get('small', {}).get(t, {}).items():
        b = tp.bidx.get(int(k))
        if b is None or not tp.bins[b][3]: continue
        tp.set_bin(b, [idx[d] for d in dids])
    bins, items = tp.bins, tp.items
    rest_small = [b for b in range(len(bins)) if b not in tp.asg and bins[b][1] <= 3000]
    pl = tp.pool(); sub = [items[j] for j in pl]
    r = match_small([bins[b] for b in rest_small], sub, lam=0.3)
    for q, b in enumerate(rest_small):
        if r[q] is None: tp.failed.add(b)
        else: tp.set_bin(b, [pl[r[q]]])
    big = [b for b in range(len(bins)) if b not in tp.asg and b not in tp.failed]
    pool = SortedList((items[j][0], j) for j in tp.pool() if items[j][2])
    pk = Packer([], tp.rng, K=300, G0=20000); pk.pool = pool
    for b in sorted([b for b in big if bins[b][3]], key=lambda b: bins[b][1]):
        k, L, U, F = bins[b]
        rr = pk.fill(L, U, 0) or pk.fill(L, U, 30)
        if rr is None: tp.failed.add(b); continue
        tp.set_bin(b, [j for _, j in rr])
    for j in tp.pool():
        if not items[j][2]: pool.add((items[j][0], j))
    for b in sorted([b for b in big if not bins[b][3]], key=lambda b: bins[b][1]):
        k, L, U, F = bins[b]
        rr = pk.fill(L, U, 0) or pk.fill(L, U, 30)
        if rr is None: tp.failed.add(b); continue
        tp.set_bin(b, [j for _, j in rr])
    return tp
def run(plan, t, seed, tmax, log):
    tp = build_tp(plan, t, seed)
    log(f't{t} init {tp.stats()}')
    t0 = time.time(); it = 0
    while tp.failed and time.time() - t0 < tmax:
        it += 1
        if it % 3 == 1:
            fr = squeeze(tp, nbins=tp.rng.choice([30, 40, 60]), tl=12)
        fl = sorted(tp.failed, key=lambda b: tp.bins[b][1])
        nf, dw = lns_step(tp, [tp.rng.choice(fl)], ndonor=tp.rng.choice([20, 30, 40, 60]), tl=10)
    log(f't{t} end {tp.stats()} {time.time()-t0:.0f}s')
    return tp
if __name__ == '__main__':
    P = json.load(open(sys.argv[1])); P = [P] if isinstance(P, dict) else P
    plan = P[int(sys.argv[2])]; tmax = float(sys.argv[3]); seed = int(sys.argv[4]); out = sys.argv[5]
    log = lambda s: print(s, flush=True)
    log(f'plan obj {plan.get("obj")}')
    tps = {}
    for t in os.environ.get('ORDER', 'itcn'):
        tps[t] = run(plan, t, seed, tmax, log)
        if tps[t].failed and os.environ.get('STOPFAIL'): log('fail stop'); sys.exit()
    dm, mm, trained, halfs, us, E = assemble(plan, tps)
    sol = {'dataMappings': [[d, m] for d, m in dm.items()], 'modelMappings': mm}
    sc = check(sol) if E <= INST['energyCap'] else -1
    log(f'assembled {sc} E-cap {E-INST["energyCap"]}')
    cp = {}
    for d, m in dm.items(): cp[m] = cp.get(m, 0) | D[d]['isCopyrighted']
    tg = {b for _, b in mm}
    h2 = {i for i in trained if cp.get(i, 0) or i in tg}; f2 = trained - h2
    try: greedy_add(INST, dm, mm, f2, h2, us)
    except Exception as ex: log(f'greedy err {ex}')
    sol2 = {'dataMappings': [[d, m] for d, m in dm.items()], 'modelMappings': mm}
    used = set(m for _, m in sol2['dataMappings']) | set(a for a, b in mm) | set(b for a, b in mm)
    E2 = sum(M[i]['energyCost'] for i in used)
    sc2 = check(sol2) if E2 <= INST['energyCap'] else -1
    best = sol2 if sc2 >= sc else sol
    json.dump(best, open(out, 'w')); log(f'FINAL {max(sc,sc2)}')
    if max(sc, sc2) > 272320:
        subprocess.run(['python3', SP+'medEnergy/saveout.py', '2_medium', os.path.abspath(out), '272320'])
