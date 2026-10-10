"""Pack a planB (with exact small assignment) into a solution. usage: packB.py plan.json out.json"""
import sys, json, os, random
sys.path.insert(0, '/home/user/EID')
from exact_pack import pack_type
from test_solution import getSolutionScore
from solve import strict_ok
INSTP = '/home/user/EID/datasets/2_medium.json'; TXT = open(INSTP).read(); INST = json.loads(TXT)
M, D = INST['models'], INST['datasets']; T = 'ntic'
OUT = '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_lb/2_medium_submission.json'


def check(sol):
    sc, ok, msg = getSolutionScore(json.dumps(sol), TXT, False)
    E = 0
    used = {m for _, m in sol['dataMappings']} | {a for a, _ in sol['modelMappings']} | {b for _, b in sol['modelMappings']}
    E = sum(M[i]['energyCost'] for i in used)
    return sc if (ok and strict_ok(INST, sol) and E <= INST['energyCap']) else -1


def save_if_better(sol, log=print):
    sc = check(sol)
    b = 272320
    if os.path.exists(OUT):
        try: b = max(b, check(json.load(open(OUT))))
        except Exception: pass
    if sc > b:
        tmp = OUT + '.tmpB%d' % os.getpid(); json.dump(sol, open(tmp, 'w')); os.replace(tmp, OUT)
        log(f'  *** wrote {sc} to out_lb (prev {b})')
    return sc


def pack(p, seeds=8, log=print):
    full, half, src = set(p['full']), set(p['half']), set(p['source'])
    srcreq = {(a, t) for a, t in p['srcreq']}
    dm = {}; used = set()
    small = {}
    for k, v in p['small'].items():
        i, t, kind = k.split(','); small[(int(i), t)] = v
        for d in v: dm[d] = int(i); used.add(d)
    fails = {}
    for t in T:
        bins = []
        for i in full:
            r = M[i]['dataRequirements'].get(t)
            if r and (i, t) not in small: bins.append((i, r['lowerBound'], r['upperBound']))
        items = [(d['size'], d['id']) for d in D if d['dataType'] == t and not d['isCopyrighted'] and d['id'] not in used]
        best = None
        for sd in range(seeds):
            order = ['lb_desc', 'width', 'random', 'lb_asc'][sd % 4]
            fill, failed, left = pack_type(bins, items, seed=sd, order=order)
            if best is None or len(failed) < len(best[1]): best = (fill, failed, left)
            if not failed: break
        fill, failed, left = best
        if failed: fails[t] = [(k, M[k]['dataRequirements'][t]['lowerBound']) for k in failed]
        for k, v in fill.items():
            for s, d in v[2]: dm[d] = k; used.add(d)
        # half + source bins (non-source reqs) with any remaining items
        hb = []
        for i in half | src:
            r = M[i]['dataRequirements'].get(t)
            if r and (i, t) not in srcreq and (i, t) not in small: hb.append((i, r['lowerBound'], r['upperBound']))
        items2 = [(d['size'], d['id']) for d in D if d['dataType'] == t and d['id'] not in used]
        fill2, failed2, left2 = pack_type(hb, items2, seed=0, order='width')
        if failed2: fails[t + 'H'] = failed2
        for k, v in fill2.items():
            for s, d in v[2]: dm[d] = k; used.add(d)
    # sources
    mm = []; pool = {t: sorted([s for s in src if t in M[s]['dataRequirements']]) for t in T}
    for a, t in sorted(srcreq):
        if not pool[t]: fails.setdefault('src', []).append((a, t)); continue
        mm.append([pool[t].pop(), a])
    sol = {'dataMappings': [[d, m] for d, m in dm.items()], 'modelMappings': mm}
    return sol, fails


if __name__ == '__main__':
    p = json.load(open(sys.argv[1]))
    sol, fails = pack(p)
    print('fails', fails)
    sc = check(sol); print('score', sc, 'plan obj', p.get('obj'))
    json.dump(sol, open(sys.argv[2], 'w'))
    if sc > 0: save_if_better(sol)
