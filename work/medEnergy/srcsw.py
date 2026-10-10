"""Swap source roles: source S1 -> trained (full/half), mono S2 -> source. usage: srcsw.py plan out mfs mts"""
import sys, json, time
import numpy as np
exec(open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/medEnergy/src8.py').read().split("for k, t in enumerate(T4):\n    z = ")[0])
t0 = time.time(); improved = True
while improved and time.time() - t0 < 400:
    improved = False; base = val(mode); best = None
    for k, t in enumerate(T4):
        S1s = np.nonzero((mode == 3) & HAS[:, k])[0]
        z = np.nonzero(MONO & HAS[:, k] & ~BAD & ((mode == 1) | (mode == 2)))[0]
        S2s = sorted(z, key=lambda s: (loss(s), LB[s, k]))[:25]
        for s1 in S1s:
            for s2 in S2s:
                for md in (1, 2):
                    g = (int(V[s1]) if md == 1 else int(V[s1] // 2)) - loss(s2)
                    if g <= 0 or (best and g <= best[0]): continue
                    o1, o2 = mode[s1], mode[s2]; mode[s1] = md; mode[s2] = 3
                    if ok(slacks(mode)): best = (g, int(s1), int(s2), md)
                    mode[s1] = o1; mode[s2] = o2
    if best:
        g, s1, s2, md = best; mode[s1] = md; mode[s2] = 3; sl = slacks(mode); improved = True
        print(f'swap src {s1}->{md}, {s2}->src gain {g} value {val(mode)} {time.time()-t0:.0f}s', flush=True); dump()
print('final', val(mode), flush=True); dump()
