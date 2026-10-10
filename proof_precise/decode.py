# usage: decode.py input.txt hitsfile... -> writes plan_<n>.json for hits sorted by (min free slack desc)
import json, sys, numpy as np
inp = sys.argv[1]
z = np.load('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/mitm2/devs_m0.npz')
sel = np.load(inp + '.sel.npy')
di, do, ok, bo = z['di'], z['do'], z['ok'], z['bestopt']
inst = json.load(open('/home/user/EID/datasets/4_precise.json')); M, D = inst['models'], inst['datasets']
cap = inst['energyCap']; TY = 'ntic'
F = {t: 0 for t in TY}; T = {t: 0 for t in TY}
for d in D:
    T[d['dataType']] += d['size']
    if not d['isCopyrighted']: F[d['dataType']] += d['size']
plans = []
for fn in sys.argv[2:]:
    for line in open(fn):
        if not line.startswith('HIT'): continue
        p = line.split(); its = [int(x) for x in p[p.index('items') + 1:]]
        opt = {int(i): int(o) for i, o in zip(ok, bo)}
        for k in its:
            j = sel[k]; opt[int(di[j])] = int(do[j])
        full = sorted(i for i, o in opt.items() if o == 1); half = sorted(i for i, o in opt.items() if o == 2)
        val = sum(M[i]['value'] for i in full) + sum(M[i]['value'] // 2 for i in half)
        E = sum(M[i]['energyCost'] for i in full + half)
        fs = {t: F[t] - sum(M[i]['dataRequirements'][t]['lowerBound'] for i in full if t in M[i]['dataRequirements']) for t in TY}
        ts = {t: T[t] - sum(M[i]['dataRequirements'][t]['lowerBound'] for i in full + half if t in M[i]['dataRequirements']) for t in TY}
        plans.append((min(fs.values()), val, cap - E, fs, ts, full, half))
plans.sort(key=lambda x: -x[0])
print(len(plans), 'plans')
for n, (mf, val, es, fs, ts, full, half) in enumerate(plans[:int(1e9)]):
    if n < 30: print(n, 'value', val, 'Eslack', es, 'free', fs, 'totmin', min(ts.values()))
    if n < 50:
        json.dump({'full': full, 'half': half, 'source': [], 'srcreq': [], 'obj': val}, open(f'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/mitm2/plan_{n}.json', 'w'))
