import json, sys, time
sys.path.insert(0, '/home/user/EID')
import numpy as np, mip4
TYPES = 'ntic'
inst = json.load(open('/home/user/EID/datasets/4_precise.json'))
M, D = inst['models'], inst['datasets']; cap = inst['energyCap']
margin = float(sys.argv[1]) if len(sys.argv) > 1 else 0
F = {t: 0 for t in TYPES}
for d in D:
    if not d['isCopyrighted']: F[d['dataType']] += d['size']
lpv, rc, ok, x = mip4.core_solve(inst, margin, 10)
n = len(ok)
def vec(i, o):
    if o == 0: return np.zeros(5, np.int64)
    e = M[i]['energyCost']
    if o == 2: return np.array([e, 0, 0, 0, 0], np.int64)
    r = M[i]['dataRequirements']
    return np.array([e] + [r[t]['lowerBound'] if t in r else 0 for t in TYPES], np.int64)
def val(i, o):
    return 0 if o == 0 else (M[i]['value'] if o == 1 else M[i]['value'] // 2)
best = {}; devs = []
for k, i in enumerate(ok):
    pf, ph = rc[k]
    opts = sorted([(pf, 1), (ph, 2), (0.0, 0)], reverse=True)
    best[i] = opts[0][1]
    for p, o in opts[1:]:
        c = opts[0][0] - p
        if c <= 6.5:
            devs.append((c, i, o))
R = np.array([cap] + [F[t] for t in TYPES], np.int64)
U0 = sum((vec(i, best[i]) for i in ok), np.zeros(5, np.int64))
V0 = sum(val(i, best[i]) for i in ok)
r0 = R - U0
# recover lam from devs exactly via lstsq on all
A = []; b = []
for c, i, o in devs:
    bo = best[i]
    dd = vec(i, o) - vec(i, bo)
    dv = val(i, o) - val(i, bo)
    # c = -(dv) + lam.dd
    A.append(dd.astype(float)); b.append(c + dv)
lam = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
res = np.array(A) @ lam - np.array(b)
print('LP', lpv, 'V0', V0, 'r0', r0.tolist(), 'lam', lam.tolist(), 'lam.r0', float(lam @ r0), 'resid', np.abs(res).max())
print('check LP = V0 + lam.r0 :', V0 + lam @ r0)
devs.sort()
dc = np.array([c for c, i, o in devs]); di = np.array([i for c, i, o in devs]); do = np.array([o for c, i, o in devs])
dd = np.array([vec(i, o) - vec(i, best[i]) for c, i, o in devs]); dv = np.array([val(i, o) - val(i, best[i]) for c, i, o in devs])
np.savez(f'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/mitm2/devs_m{int(margin)}.npz', dc=dc, di=di, do=do, dd=dd, dv=dv, r0=r0, lam=lam, V0=V0, lpv=lpv, R=R,
         ok=np.array(ok), bestopt=np.array([best[i] for i in ok]))
for th in [0.5, 1, 2, 3, 4, 5, 5.93]:
    print(th, (dc <= th).sum(), 'models', len(set(di[dc <= th])))
from collections import Counter
print('kinds', Counter((best[i], o) for c, i, o in devs if c <= 5.93))
print('Energy deltas abs (cost<=5.93):', np.percentile(np.abs(dd[dc <= 5.93, 0]), [0, 25, 50, 75, 100]))
for j in range(1, 5):
    print(TYPES[j-1], np.percentile(np.abs(dd[dc <= 5.93, j]), [0, 25, 50, 75, 100]))
