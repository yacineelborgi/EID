"""Source retargeting + adds on top of swap5 machinery.
usage: src5.py plan.json out.json mf mt"""
import sys, json, time
import numpy as np
sys.argv = sys.argv[:5] + ['0']
exec(open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/medEnergy/swap5.py').read().split("t0 = time.time()")[0])
MONO = HAS.sum(1) == 1
BADT = np.zeros((nM, 4), bool)
for i, m in enumerate(M):
    for t, r in m['dataRequirements'].items():
        if r['upperBound'] < r['lowerBound']: BADT[i, T4.index(t)] = True
def retarget():
    for k, t in enumerate(T4):
        ns = int(((mode == 3) & MONO & HAS[:, k]).sum())
        must = np.nonzero((mode == 2) & BADT[:, k])[0]
        opt = np.nonzero((mode == 2) & HAS[:, k] & ~BADT[:, k])[0]
        opt = opt[np.argsort(-LB[opt, k])]
        before = int(LB[SRQ[:, k] & (mode == 2), k].sum())
        SRQ[:, k] = False; SRQ[must, k] = True
        rest = ns - len(must); assert rest >= 0, (t, ns, len(must))
        SRQ[opt[:rest], k] = True
        after = int(LB[SRQ[:, k] & ~BADT[:, k], k].sum()) + int(LB[must, k].sum())
        print(f'  retarget {t}: sources {ns} must {len(must)} sourced lb {before} -> {after}', flush=True)
def dump():
    p = {'full': np.nonzero(mode == 1)[0].tolist(), 'half': np.nonzero(mode == 2)[0].tolist(),
         'source': np.nonzero(mode == 3)[0].tolist(), 'srcreq': [[int(i), T4[k]] for i, k in zip(*np.nonzero(SRQ)) if mode[i] == 2]}
    p['obj'] = val(mode); json.dump(p, open(out, 'w'))
retarget(); sl = slacks(mode)
print('after retarget feasible', ok(sl), {t: (int(sl[t][0].min()), int(sl[t][1].min())) for t in T4}, val(mode), flush=True)
# adds: untrained good models as half / full
improved = True
while improved:
    improved = False
    offs = np.nonzero((mode == 0) & (V > 0))[0]
    cand = []
    for i in offs:
        if BAD[i]: continue
        cand.append((V[i], i, 1)); cand.append((V[i] // 2, i, 2))
    cand.sort(key=lambda c: -c[0])
    for g, i, md in cand:
        if g <= 0: break
        mode[i] = md; s2 = slacks(mode)
        if ok(s2):
            sl = s2; improved = True; print(f'  add {i} as {md} gain {g} value {val(mode)}', flush=True); retarget(); sl = slacks(mode); break
        mode[i] = 0
print('final', val(mode), ok(sl), {t: (int(sl[t][0].min()), int(sl[t][1].min())) for t in T4}, flush=True)
dump()
