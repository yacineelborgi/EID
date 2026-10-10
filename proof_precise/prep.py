# usage: prep.py target Lfree out.txt [ratio] [synthetic_seed]
import numpy as np, sys
z = np.load('/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/mitm2/devs_m0.npz')
dc, di, dd, dv, r0, lam, V0 = z['dc'], z['di'], z['dd'], z['dv'], z['r0'].copy(), z['lam'], int(z['V0'])
target = int(sys.argv[1]); Lf = int(sys.argv[2]) if ',' not in sys.argv[2] else 0; out = sys.argv[3]
ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
L = np.array([0, Lf, Lf, Lf, Lf], np.int64) if ',' not in sys.argv[2] else np.array([int(x) for x in sys.argv[2].split(',')], np.int64)
dvneed = target - V0
if len(sys.argv) > 5:  # synthetic test: plant a subset
    rng = np.random.default_rng(int(sys.argv[5]))
    S = []; used = set(); c = 0
    for k in rng.permutation(len(dc)):
        if di[k] in used or c + dc[k] > 2.2: continue
        if rng.random() < 0.3: S.append(k); used.add(di[k]); c += dc[k]
    tot = dd[S].sum(0)
    r0 = tot + L + np.array([300_000_000, 10, 20, 5, 30])
    dvneed = int(dv[S].sum()) - 1
    V0 = target - dvneed
    print('planted', sorted(S), 'cost', c)
Bud0 = float(lam @ r0) - dvneed  # = LP - target
Bud = Bud0 - float(lam @ L)
sel = np.nonzero(dc <= Bud + 1e-9)[0]
print('Bud0', Bud0, 'Bud', Bud, 'items', len(sel))
# groups by model; split groups alternately by min cost (balanced cost mass), ratio tweak
gm = {}
for k in sel: gm.setdefault(int(di[k]), []).append(int(k))
gl = sorted(gm, key=lambda m: min(dc[k] for k in gm[m]))
side = {}
acc = [0.0, 0.0]
for m in gl:
    w = min(dc[k] for k in gm[m])
    s = 0 if acc[0] * (1 - ratio) <= acc[1] * ratio else 1
    side[m] = s; acc[s] += max(w, 0.05)
with open(out, 'w') as f:
    f.write(' '.join(str(int(x)) for x in r0) + '\n')
    f.write(' '.join(repr(float(x)) for x in lam) + '\n')
    f.write(' '.join(str(int(x)) for x in L) + '\n')
    f.write(f'{dvneed} {Bud!r} {Bud/2!r} {len(sel)}\n')
    for k in sel:
        f.write(f'{int(di[k])} {side[int(di[k])]} {float(dc[k])!r} {int(dv[k])} ' + ' '.join(str(int(x)) for x in dd[k]) + '\n')
# map file: line index -> deviation index
np.save(out + '.sel.npy', sel)
