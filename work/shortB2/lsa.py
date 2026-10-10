import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA')
from ptype import type_data
plan = json.load(open(sys.argv[1])); TH = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
for t in 'ntic':
    bins, items = type_data(plan, t)
    sz = np.array([s for s, _, _ in items], float); fr = np.array([f for _, _, f in items])
    small = [b for b in bins if b[1] <= TH]; big = [b for b in bins if b[1] > TH]
    C = np.full((len(small), len(items)), 1e9)
    for q, (k, L, U, F) in enumerate(small):
        ok = (sz >= L) & (sz <= U)
        if F: ok &= fr
        C[q, ok] = sz[ok] - L
    r, c = linear_sum_assignment(C)
    w = C[r, c]; bad = (w >= 1e9).sum(); w = w[w < 1e9].sum()
    used = set(c.tolist())
    left = sum(sz[j] for j in range(len(items)) if j not in used)
    leftF = sum(sz[j] for j in range(len(items)) if j not in used and fr[j])
    bigL = sum(b[1] for b in big); bigLF = sum(b[1] for b in big if b[3])
    al = sorted(sz); Ls = sorted(b[1] for b in bins); G = sum(max(a, b) for a, b in zip(Ls, al)) - sum(Ls)
    print(t, 'small', len(small), 'unmatched', bad, 'waste', int(w), 'G-sumL', int(G), 'big', len(big), 'slack big', int(left - bigL), 'F slack big', int(leftF - bigLF))
