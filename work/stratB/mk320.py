import json, numpy as np, exm, sys
sys.path.insert(0,'/home/user/EID'); from exact_pack import dp_min_subset
from collections import defaultdict
mo = exm.build(marg={t:10 for t in 'ntic'}); M=exm.M
S = json.load(open('../out_lb/2_medium_submission.json'))
full, half, src = exm.sol_status(S)
cur = {i: ('f' if i in full else 'h' if i in half else 's' if i in src else '-') for i in range(len(M))}
x0 = np.zeros(len(mo.cost))
for i, st_ in cur.items():
    d = {'f': mo.fv, 'h': mo.hv, 's': mo.sv}.get(st_)
    if d is not None: x0[d[i]] = 1
asg = defaultdict(list)
for dd, m in S['dataMappings']: asg[(m, exm.D[dd]['dataType'])].append(dd)
col = {(b, did): k for k, b, did, sz in mo.xinfo}
for (i, t, kind) in mo.sbins:
    if not ((kind == 'F' and cur[i] == 'f') or (kind == 'H' and cur[i] in 'hs' and t not in mo.unf[i])): continue
    for d in asg[(i, t)]: x0[col[((i, t, kind), d)]] = 1
p = exm.plan_from_x(mo, x0); print(p['obj']); json.dump(p, open('plan320.json', 'w'))
