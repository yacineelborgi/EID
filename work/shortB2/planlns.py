"""Plan LNS over mip6 margin model. usage: planlns.py start_plan mA mF T total_s out_prefix seed [nfree] [tl]"""
import json, sys, time, random
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortB')
import mip6, highspy
from enc import encode_plan
inst = json.load(open('/home/user/EID/datasets/6_shortage.json'))
M = inst['models']
sp, MG, T, tot, outp, seed = sys.argv[1], json.loads(sys.argv[2]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], int(sys.argv[7])
nfree = int(sys.argv[8]) if len(sys.argv) > 8 else 2500
tl = float(sys.argv[9]) if len(sys.argv) > 9 else 40
rng = random.Random(seed)
marg = {}
for t in 'ntic':
    marg[('A', t)] = MG[t][0]; marg[('F', t)] = MG[t][1]
t0 = time.time()
mo = mip6.build(inst, marg=marg)
h = mip6.to_highs(mo, integer=False); h.run()
lpv = h.getInfo().objective_function_value
s = h.getSolution(); y = np.array(s.row_dual); xl = np.array(s.col_value)
rc = mo.cost - mo.A.T @ y
if T < 0:
    T = lpv + T
gap = lpv - T
fix0 = (xl[:mo.nV] < 1e-9) & (rc[:mo.nV] < -(gap + 0.5)) & (mo.ub[:mo.nV] > 0)
fix1 = (xl[:mo.nV] > 1 - 1e-9) & (rc[:mo.nV] > (gap + 0.5))
print("LP", lpv, "gap", gap, "fix0", fix0.sum(), "fix1", fix1.sum(), flush=True)
plan = json.load(open(sp))
x0 = encode_plan(mo, inst, plan)
Ax = mo.A @ x0
viol = np.maximum(mo.rlo - Ax, 0) + np.maximum(Ax - mo.rhi, 0)
v0 = mo.cost @ x0
print("start", v0, "maxviol", viol.max(), mo.names[int(viol.argmax())], flush=True)
nM = mo.nM
# model -> var indices
mvars = [[mo.fo + i, mo.ho + i, mo.so + i] for i in range(nM)]
for k, (i, t, L, isb) in enumerate(mo.zlist):
    mvars[i].append(mo.zo + k)
# candidate models: not all vars rc-fixed to 0
fx0 = np.zeros(mo.nV, bool); fx0[:] = fix0
cand = [i for i in range(nM) if not all(fx0[j] for j in mvars[i][:3])]
chosen = [i for i in range(nM) if x0[mo.fo + i] + x0[mo.ho + i] + x0[mo.so + i] > 0.5]
print("cand", len(cand), "chosen", len(chosen), flush=True)
best = v0; it = 0
while time.time() - t0 < tot:
    it += 1
    # neighbourhood: random subset of candidates, biased to |rc| small, plus related by type
    mode = rng.choice(['rand', 'type', 'rc'])
    if mode == 'type':
        tt = rng.sample('ntic', 2)
        pool = [i for i in cand if all(t in M[i]['dataRequirements'] for t in tt[:1])]
    else:
        pool = cand
    if mode == 'rc':
        sc = {i: max(rc[j] for j in mvars[i][:3] if mo.ub[j] > 0) if any(mo.ub[j] > 0 for j in mvars[i][:3]) else -1e9 for i in pool}
        pool = sorted(pool, key=lambda i: -abs(sc[i]) * rng.random())
        free = set(pool[-nfree:])
    else:
        free = set(rng.sample(pool, min(nfree, len(pool))))
    lb = mo.lb.copy(); ub = mo.ub.copy()
    ub[:mo.nV][fx0] = 0
    for i in range(nM):
        if i in free:
            continue
        for j in mvars[i]:
            lb[j] = ub[j] = x0[j]
    mo2 = mip6.Model(); mo2.__dict__.update(mo.__dict__); mo2.lb = lb; mo2.ub = ub
    for j in range(mo.nV):
        if x0[j] > ub[j] + 1e-9 or x0[j] < lb[j] - 1e-9:
            lb[j] = min(lb[j], x0[j]); ub[j] = max(ub[j], x0[j])
    hh = mip6.to_highs(mo2, integer=True)
    hh.setOptionValue('time_limit', tl)
    hh.setOptionValue('mip_rel_gap', 1e-7)
    hh.setOptionValue('random_seed', rng.randrange(1000))
    so = highspy.HighsSolution(); so.col_value = list(x0); so.value_valid = True
    hh.setSolution(so)
    hh.run()
    x = np.array(hh.getSolution().col_value)
    v = hh.getInfo().objective_function_value if len(x) == mo.nTot else -1
    print(f"it {it} mode {mode} nfree {len(free)} obj {v:.0f} bnd {hh.getInfo().mip_dual_bound:.1f} best {best} t={time.time()-t0:.0f}", flush=True)
    if len(x) == mo.nTot and v > best + 0.5:
        x0 = np.round(x * 1.0, 9)
        xi = np.round(x[:mo.nV]); x0[:mo.nV] = xi
        best = v
        p = mip6.plan_from_x(mo, x)
        p['obj'] = v; p['marg'] = MG
        json.dump(p, open(outp + '_plan.json', 'w'))
        print("  improved ->", v, flush=True)
