"""Exact-delta pair/triple swap search on 5_energy relaxation (per-type margins).
usage: swap7.py plan.json out.json mfs mts tmax"""
import sys, json, time
import numpy as np
tmax = float(sys.argv[5]); sys.argv = sys.argv[:5] + ['0']
exec(open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/medEnergy/swap5.py').read().split("t0 = time.time()")[0])
def vec(sl): return np.array([sl[t][0].min() for t in T4] + [sl[t][1].min() for t in T4], np.int64)
def dump():
    p = {'full': np.nonzero(mode == 1)[0].tolist(), 'half': np.nonzero(mode == 2)[0].tolist(),
         'source': np.nonzero(mode == 3)[0].tolist(), 'srcreq': [[int(i), T4[k]] for i, k in zip(*np.nonzero(SRQ)) if mode[i] == 2]}
    p['obj'] = val(mode); json.dump(p, open(out, 'w'))
t0 = time.time()
while time.time() - t0 < tmax:
    base = vec(sl)
    fullA = np.nonzero(mode == 1)[0]; halfB = np.nonzero((mode == 2) & ~BAD & ~SRQ.any(1))[0]
    offC = np.nonzero((mode == 0) & ~BAD & (V > 0))[0]
    dA = {}; dB = {}
    for a in fullA:
        mode[a] = 2; dA[int(a)] = vec(slacks(mode)) - base; mode[a] = 1
    for b in halfB:
        mode[b] = 1; dB[int(b)] = (vec(slacks(mode)) - base, int(G_UP[b])); mode[b] = 2
    for c in offC:
        for md, g in ((1, int(V[c])), (2, int(V[c] // 2))):
            mode[c] = md; dB[(int(c), md)] = (vec(slacks(mode)) - base, g); mode[c] = 0
    print(f'deltas computed {time.time()-t0:.0f}s; base {base.tolist()}', flush=True)
    # candidate moves: up to 2 B's with up to 1 A
    As = sorted(dA, key=lambda a: G_UP[a])
    Bs = sorted(dB, key=lambda b: -dB[b][1])
    cands = []
    Bl = [(b, dB[b][0], dB[b][1]) for b in Bs if dB[b][1] > 0]
    Al = [(a, dA[a], int(G_UP[a])) for a in As]
    AV = np.array([x[1] for x in Al]); AG = np.array([x[2] for x in Al])
    for b, d, g in Bl[:2500]:
        if (base + d >= 0).all(): cands.append((g, [b], []))
        m = ((base + d + AV) >= 0).all(1) & (AG < g)
        for q in np.nonzero(m)[0][:3]: cands.append((g - AG[q], [b], [Al[q][0]]))
    # pairs of B with one A
    top = Bl[:300]
    for x in range(len(top)):
        for y in range(x + 1, len(top)):
            b1, d1, g1 = top[x]; b2, d2, g2 = top[y]
            if b1 == b2 or (isinstance(b1, tuple) and isinstance(b2, tuple) and b1[0] == b2[0]): continue
            dd = base + d1 + d2
            m = ((dd + AV) >= 0).all(1) & (AG < g1 + g2)
            if m.any():
                q = np.nonzero(m)[0][0]; cands.append((g1 + g2 - AG[q], [b1, b2], [Al[q][0]]))
    cands.sort(key=lambda c: -c[0])
    print(f'cands {len(cands)} top {[c[0] for c in cands[:5]]} {time.time()-t0:.0f}s', flush=True)
    applied = False
    for g, bs, as_ in cands[:300]:
        snap = mode.copy()
        for a in as_: mode[a] = 2
        for b in bs:
            if isinstance(b, tuple): mode[b[0]] = b[1]
            else: mode[b] = 1
        s2 = slacks(mode)
        if ok(s2) and val(mode) > val(snap):
            sl = s2; applied = True; print(f'  applied gain {val(mode)-val(snap)} B {bs} A {as_} value {val(mode)}', flush=True); dump(); break
        mode[:] = snap
    if not applied: print('no move', flush=True); break
print('final', val(mode), vec(sl).tolist(), flush=True); dump()
