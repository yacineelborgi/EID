"""Add untrained BAD models as half with their bad reqs sourced by newly created sources (cheapest monos).
usage: src8.py plan.json out.json mfs mts"""
import sys, json, time
import numpy as np
sys.argv = sys.argv[:5] + ['0']
exec(open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/medEnergy/swap5.py').read().split("t0 = time.time()")[0])
MONO = HAS.sum(1) == 1
BADT = np.zeros((nM, 4), bool)
for i, m in enumerate(M):
    for t, r in m['dataRequirements'].items():
        if r['upperBound'] < r['lowerBound']: BADT[i, T4.index(t)] = True
def dump():
    p = {'full': np.nonzero(mode == 1)[0].tolist(), 'half': np.nonzero(mode == 2)[0].tolist(),
         'source': np.nonzero(mode == 3)[0].tolist(), 'srcreq': [[int(i), T4[k]] for i, k in zip(*np.nonzero(SRQ)) if mode[i] == 2]}
    p['obj'] = val(mode); json.dump(p, open(out, 'w'))
def loss(s): return int(V[s]) if mode[s] == 1 else int(V[s] // 2) if mode[s] == 2 else 0
for k, t in enumerate(T4):
    z = np.nonzero(MONO & HAS[:, k] & ~BAD & (mode != 3))[0]
    ls = sorted((loss(s), int(LB[s, k]), int(s), int(mode[s])) for s in z)[:6]
    print(t, 'cheapest potential sources (loss, lb, id, mode)', ls, flush=True)
improved = True
while improved:
    improved = False
    Bs = [b for b in np.nonzero((mode == 0) & BAD & (V > 1))[0]]
    Bs.sort(key=lambda b: -V[b])
    for b in Bs:
        bt = [k for k in range(4) if BADT[b, k]]
        if not all(HAS[b, k] for k in bt): continue
        snap = (mode.copy(), SRQ.copy()); g = int(V[b] // 2); okk = True
        for k in bt:
            z = np.nonzero(MONO & HAS[:, k] & ~BAD & (mode != 3) & (np.arange(nM) != b))[0]
            if len(z) == 0: okk = False; break
            cl = sorted(z, key=lambda s: (loss(s), LB[s, k]))
            # pick cheapest whose conversion keeps feasibility
            s = cl[0]; g -= loss(s); mode[s] = 3; SRQ[b, k] = True
        mode[b] = 2
        if okk and g > 0:
            s2 = slacks(mode)
            if ok(s2):
                sl = s2; improved = True; print(f'  add bad {b} v {V[b]} gain {g} value {val(mode)}', flush=True); dump(); break
        mode[:] = snap[0]; SRQ[:] = snap[1]
print('final', val(mode), {t: (int(sl[t][0].min()), int(sl[t][1].min())) for t in T4}, flush=True); dump()
