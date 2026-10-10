"""Relaxation-level swap local search for 5_energy plans (exact free/total slacks per tau, margins mf/mt).
Moves: upgrade half->full, add untrained as full/half, swap (A full->half, B half->full).
usage: swap5.py plan.json out.json mf mt [rounds]"""
import json, sys, time
import numpy as np
inst = json.load(open('/home/user/EID/datasets/5_energy.json')); M = inst['models']; D = inst['datasets']
TAUS = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)
T4 = "ntic"; nM = len(M)
plan = json.load(open(sys.argv[1])); out = sys.argv[2]; MFS = [int(x) for x in sys.argv[3].split(',')]; MTS = [int(x) for x in sys.argv[4].split(',')]
MFS = MFS*4 if len(MFS)==1 else MFS; MTS = MTS*4 if len(MTS)==1 else MTS
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
Fs = {t: np.array(sorted(d['size'] for d in D if d['dataType'] == t and not d['isCopyrighted']), np.int64) for t in T4}
Cs = {t: np.array(sorted(d['size'] for d in D if d['dataType'] == t and d['isCopyrighted']), np.int64) for t in T4}
PF = {t: np.concatenate([[0], np.cumsum(Fs[t])]) for t in T4}
PC = {t: np.concatenate([[0], np.cumsum(Cs[t])]) for t in T4}
SF = {t: int(PF[t][-1]) for t in T4}; SA = {t: int(PF[t][-1] + PC[t][-1]) for t in T4}
LB = np.zeros((nM, 4), np.int64); HAS = np.zeros((nM, 4), bool); BAD = np.zeros(nM, bool)
for i, m in enumerate(M):
    for t, r in m['dataRequirements'].items():
        k = T4.index(t); LB[i, k] = r['lowerBound']; HAS[i, k] = True
        if r['upperBound'] < r['lowerBound']: BAD[i] = True
V = np.array([m['value'] for m in M]); G_UP = V - V // 2
mode = np.zeros(nM, int)  # 0 off,1 full,2 half,3 src
for i in plan['full']: mode[i] = 1
for i in plan['half']: mode[i] = 2
for i in plan['source']: mode[i] = 3
SRQ = np.zeros((nM, 4), bool)
for i, t in plan['srcreq']: SRQ[i, T4.index(t)] = True
def Gf(t, kf, kh):
    F, C = Fs[t], Cs[t]
    lo, hi = max(0, kh - len(C)), min(kh, len(F) - kf)
    if lo > hi: return 10**15
    # f(a) = PF[kf+a]-PF[kf] + PC[kh-a], convex: find a where F[kf+a] >= C[kh-a-1]
    while lo < hi:
        a = (lo + hi) // 2
        if F[kf + a] >= C[kh - a - 1]: hi = a
        else: lo = a + 1
    return int(PF[t][kf + lo] + PC[t][kh - lo])
def slacks(mode):
    """returns dict t -> (array free slack per tau, array tot slack per tau)"""
    res = {}
    for k, t in enumerate(T4):
        f = (mode == 1) & HAS[:, k]
        h = ((mode == 2) & HAS[:, k] & ~SRQ[:, k]) | ((mode == 3) & HAS[:, k])
        lf = LB[f, k]; lh = LB[h, k]
        fs, ts = [], []
        for tau in TAUS:
            kf = int((lf <= tau).sum()); kh = int((lh <= tau).sum())
            bigF = int(lf[lf > tau].sum()); bigA = bigF + int(lh[lh > tau].sum())
            fs.append(SF[t] - bigF - int(PF[t][kf]) - MFS[k] if kf <= len(Fs[t]) else -10**9)
            ts.append(SA[t] - bigA - Gf(t, kf, kh) - MTS[k])
        res[t] = (np.array(fs), np.array(ts))
    return res
def ok(sl): return all(sl[t][0].min() >= 0 and sl[t][1].min() >= 0 for t in T4)
def val(mode): return int(V[mode == 1].sum() + (V[mode == 2] // 2).sum())
sl = slacks(mode); print('start value', val(mode), 'feasible', ok(sl), {t: (int(sl[t][0].min()), int(sl[t][1].min())) for t in T4}, flush=True)
assert ok(sl)
t0 = time.time()
for rnd in range(rounds):
    avail = np.array([sl[t][0].min() for t in T4])
    # candidates B: half, no srcreq, not bad; untrained good models
    halfB = np.nonzero((mode == 2) & ~BAD & ~SRQ.any(1))[0]
    fullA = np.nonzero(mode == 1)[0]
    best = None
    # single upgrades / adds approximated with big-bin deltas, verified exactly
    cands = []
    ok1 = (LB[halfB] <= avail).all(1)
    for b in halfB[ok1]: cands.append((int(G_UP[b]), [(int(b), 1)]))
    LA = LB[fullA]; gA = G_UP[fullA]
    order = halfB[np.argsort(-G_UP[halfB])]
    for b in order[:3000]:
        need = LB[b] - avail  # need LA >= need for all types
        m = (LA >= need).all(1) & (gA < G_UP[b])
        if m.any():
            idx = np.nonzero(m)[0]
            idx = idx[np.argsort(gA[idx])][:5]
            for q in idx: cands.append((int(G_UP[b] - gA[q]), [(int(fullA[q]), 2), (int(b), 1)]))
    cands.sort(key=lambda c: -c[0])
    applied = False
    for g, mv in cands[:400]:
        if g <= 0: break
        old = [(i, mode[i]) for i, _ in mv]
        for i, md in mv: mode[i] = md
        s2 = slacks(mode)
        if ok(s2):
            sl = s2; applied = True
            print(f'rnd {rnd} gain {g} move {mv} value {val(mode)} {time.time()-t0:.0f}s', flush=True)
            break
        for i, md in old: mode[i] = md
    if not applied:
        print('no improving move', flush=True); break
    p = {'full': np.nonzero(mode == 1)[0].tolist(), 'half': np.nonzero(mode == 2)[0].tolist(),
         'source': np.nonzero(mode == 3)[0].tolist(), 'srcreq': [[int(i), T4[k]] for i, k in zip(*np.nonzero(SRQ)) if mode[i] == 2]}
    p['obj'] = val(mode); json.dump(p, open(out, 'w'))
print('final', val(mode), {t: (int(sl[t][0].min()), int(sl[t][1].min())) for t in T4})
