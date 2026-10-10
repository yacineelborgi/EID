import json, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA')
from ptype import type_data
TH = 10000
for f in sys.argv[1:]:
    plan = json.load(open(f)); out = []
    for t in 'ntic':
        bins, items = type_data(plan, t)
        sz = np.array([s for s, _, _ in items], float); fr = np.array([f_ for _, _, f_ in items])
        small = [b for b in bins if b[1] <= TH]; big = [b for b in bins if b[1] > TH]
        C = np.full((len(small), len(items)), 1e12)
        for q, (k, L, U, F) in enumerate(small):
            ok = (sz >= L) & (sz <= U)
            if F:
                ok &= fr; C[q, ok] = sz[ok] - L
            else:
                C[q, ok] = sz[ok] - L + 1e6 * fr[ok]
        r, c = linear_sum_assignment(C)
        used = set(int(j) for q, j in zip(r, c) if C[q, j] < 1e12)
        bad = sum(1 for q, j in zip(r, c) if C[q, j] >= 1e12)
        left = sum(sz[j] for j in range(len(items)) if j not in used)
        leftF = sum(sz[j] for j in range(len(items)) if j not in used and fr[j])
        bigL = sum(b[1] for b in big); bigLF = sum(b[1] for b in big if b[3])
        out.append(f"{t}: A {int(left - bigL)} F {int(leftF - bigLF)} bad {bad}")
    print(f, plan.get('obj'), ' | '.join(out), flush=True)
