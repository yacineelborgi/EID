"""src5 + source create/release moves with adds. usage: src6.py plan.json out.json mf mt"""
import sys, json, time
import numpy as np
exec(open('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/medEnergy/src5.py').read().split("retarget(); sl = slacks(mode)")[0])
def snapshot(): return mode.copy(), SRQ.copy()
def restore(s):
    mode[:] = s[0]; SRQ[:] = s[1]
def try_adds():
    """greedily add untrained good models; returns gain"""
    g0 = val(mode)
    while True:
        offs = [i for i in np.nonzero((mode == 0) & (V > 0) & ~BAD)[0]]
        cand = sorted([(V[i], i, 1) for i in offs] + [(V[i] // 2, i, 2) for i in offs], key=lambda c: -c[0])
        done = False
        for g, i, md in cand:
            if g <= 0: break
            mode[i] = md
            if ok(slacks(mode)): done = True; break
            mode[i] = 0
        if not done: return val(mode) - g0
retarget(); sl = slacks(mode); try_adds(); retarget()
print('start', val(mode), ok(slacks(mode)), flush=True)
t0 = time.time()
while time.time() - t0 < float(__import__('os').environ.get('TMAX', '240')):
    base = val(mode); best = None
    # M2: release sources
    for s in np.nonzero(mode == 3)[0]:
        for md in (1, 2):
            snap = snapshot(); mode[s] = md
            try:
                retarget_q = True
                for k in range(4):
                    pass
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()): retarget()
            except AssertionError:
                restore(snap); continue
            if ok(slacks(mode)):
                with contextlib.redirect_stdout(io.StringIO()): try_adds(); retarget()
                g = val(mode) - base
                if g > 0 and (best is None or g > best[0]): best = (g, 'release', int(s), md, snapshot())
            restore(snap)
    # M1: create a source from a cheap mono model, then adds
    for k, t in enumerate(T4):
        monos = np.nonzero(MONO & HAS[:, k] & ~BAD & ((mode == 1) | (mode == 2)))[0]
        loss = np.where(mode[monos] == 1, V[monos], V[monos] // 2)
        for s in monos[np.argsort(loss)][:15]:
            snap = snapshot(); mode[s] = 3
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()): retarget()
            if ok(slacks(mode)):
                with contextlib.redirect_stdout(io.StringIO()): try_adds(); retarget()
                g = val(mode) - base
                if ok(slacks(mode)) and g > 0 and (best is None or g > best[0]): best = (g, 'create', int(s), t, snapshot())
            restore(snap)
    if best is None: print('no move', flush=True); break
    restore(best[4]); print(f'move {best[:4]} value {val(mode)} {time.time()-t0:.0f}s', flush=True); dump()
sl = slacks(mode); print('final', val(mode), ok(sl), {t: (int(sl[t][0].min()), int(sl[t][1].min())) for t in T4}, flush=True)
dump()
