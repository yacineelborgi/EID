"""Trim a plan so that per-type assortative consumption bounds leave margins (mA all-pool, mF free-pool)."""
import json, sys
inst = json.load(open('/home/user/EID/datasets/6_shortage.json')); M, D = inst['models'], inst['datasets']
plan = json.load(open(sys.argv[1])); mA = int(sys.argv[2]); mF = int(sys.argv[3]); out = sys.argv[4]
al = {t: sorted(d['size'] for d in D if d['dataType'] == t) for t in 'ntic'}
fs = {t: sorted(d['size'] for d in D if d['dataType'] == t and not d['isCopyrighted']) for t in 'ntic'}
full = set(plan['full']); half = set(plan['half']); src = list(plan['source'])
srcreq = [tuple(x) for x in plan['srcreq']]
def G(L, items):
    return sum(max(a, b) for a, b in zip(sorted(L), items)) + (10**12 if len(L) > len(items) else 0)
def state():
    sr = set(srcreq)
    ov = {}
    for t in 'ntic':
        LA = [M[i]['dataRequirements'][t]['lowerBound'] for i in full | half if t in M[i]['dataRequirements'] and (i, t) not in sr]
        LA += [M[i]['dataRequirements'][t]['lowerBound'] for i in src if t in M[i]['dataRequirements']]
        LF = [M[i]['dataRequirements'][t]['lowerBound'] for i in full if t in M[i]['dataRequirements']]
        ov[('A', t)] = G(LA, al[t]) - (sum(al[t]) - mA)
        ov[('F', t)] = G(LF, fs[t]) - (sum(fs[t]) - mF)
    return ov
ov = state(); print('initial over', ov)
removed = 0; lost = 0
while any(v > 0 for v in ov.values()):
    viol = {k for k, v in ov.items() if v > 0}
    sr = set(srcreq)
    best = None
    for i in full | half:
        val = M[i]['value'] if i in full else M[i]['value'] // 2
        gain = 0
        for t, r in M[i]['dataRequirements'].items():
            if (i, t) in sr: continue
            L = r['lowerBound']
            if ('A', t) in viol: gain += min(L, ov[('A', t)])
            if i in full and ('F', t) in viol: gain += min(L, ov[('F', t)])
        # removing a half target also frees its sources (their bins)
        if gain <= 0: continue
        ratio = val / gain
        if best is None or ratio < best[0]: best = (ratio, i, val)
    _, i, val = best
    full.discard(i); half.discard(i); lost += val; removed += 1
    # drop sources matched to its sourced requirements (one per type)
    for (j, t) in [x for x in srcreq if x[0] == i]:
        srcreq.remove((j, t))
        cand = [s for s in src if t in M[s]['dataRequirements']]
        cand.sort(key=lambda s: -M[s]['dataRequirements'][t]['lowerBound'])
        src.remove(cand[0])
    ov = state()
print('removed', removed, 'lost', lost, 'final over', ov)
p = {'full': sorted(full), 'half': sorted(half), 'source': src, 'srcreq': [list(x) for x in srcreq], 'obj': plan['obj'] - lost}
json.dump(p, open(out, 'w'))
