"""LNS on exact-small-bin plan MIP. usage: lnsB.py tmax seed [margins n,t,i,c] [start_solution]"""
import sys, json, time, random, os, numpy as np, highspy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exm
from exm import M, T
tmax = float(sys.argv[1]); seed = int(sys.argv[2]); rng = random.Random(seed)
mg = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [10] * 4
startf = sys.argv[4] if len(sys.argv) > 4 else '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_lb/2_medium_submission.json'
NT = int(os.environ.get('NT', '60')); NU = int(os.environ.get('NU', '60')); TL = float(os.environ.get('TL', '40'))
TAG = os.environ.get('TAG', 'a')
log = lambda s: print(s, flush=True)
t0 = time.time()
mo = exm.build(marg=dict(zip(T, mg)))
nC = len(mo.cost); log(f'built cols {nC} rows {mo.A.shape[0]} {time.time()-t0:.0f}s')
h = exm.highs(mo)
h.setOptionValue('mip_rel_gap', 0.0); h.setOptionValue('random_seed', seed)
if startf.endswith('.plan'):
    p = json.load(open(startf)); full, half, src = set(p['full']), set(p['half']), set(p['source'])
else:
    full, half, src = exm.sol_status(json.load(open(startf)))
cur = {}
for i in range(len(M)):
    cur[i] = 'f' if i in full else 'h' if i in half else 's' if i in src else '-'
XB = np.array([k for k, b, did, sz in mo.xinfo]); XM = [b for k, b, did, sz in mo.xinfo]
NB0 = NB = int(os.environ.get('NB', '40')); SZ = 1.0
def fixvals(free):
    global NB
    lb = mo.lb.copy(); ub = mo.ub.copy()
    if XFIX is not None:
        fb = set(rng.sample(mo.sbins, NB)) if FT_ is None else {b for b in mo.sbins if b[1] in FT_}
        for q, k in enumerate(XB):
            b = XM[q]
            if b[0] not in free and b not in fb:
                lb[k] = ub[k] = XFIX[k]
    for i, st in cur.items():
        if i in free: continue
        for d, c in ((mo.fv, 'f'), (mo.hv, 'h'), (mo.sv, 's')):
            if i in d:
                v = 1.0 if st == c else 0.0
                lb[d[i]] = v; ub[d[i]] = v
    return lb, ub
XFIX = None; FT_ = None
E1 = [i for i in range(len(M)) if M[i]['energyCost'] <= int(os.environ.get('EMAX', '1000'))]
MODE = os.environ.get('MODE', 'rand')
def solve(free, tl, x0=None):
    lb, ub = fixvals(free)
    h.changeColsBounds(nC, np.arange(nC, dtype=np.int32), lb, ub)
    h.setOptionValue('time_limit', tl)
    if x0 is not None:
        s = highspy.HighsSolution(); s.col_value = list(map(float, x0)); s.value_valid = True; h.setSolution(s)
    h.run()
    st = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    if info.primal_solution_status != 2: return st, None, None
    return st, np.array(h.getSolution().col_value), info.mip_dual_bound
def warm(S):
    sys.path.insert(0, '/home/user/EID'); from exact_pack import dp_min_subset
    from collections import defaultdict
    x0 = np.zeros(nC)
    for i, st_ in cur.items():
        d = {'f': mo.fv, 'h': mo.hv, 's': mo.sv}.get(st_)
        if d is not None: x0[d[i]] = 1
    asg = defaultdict(list)
    for dd, m in S['dataMappings']: asg[(m, exm.D[dd]['dataType'])].append(dd)
    col = {}; cands = defaultdict(list)
    for k, b, did, sz in mo.xinfo: col[(b, did)] = k; cands[b].append((sz, did))
    used = set(); bad = []
    for (i, t, kind) in mo.sbins:
        if not ((kind == 'F' and cur[i] == 'f') or (kind == 'H' and cur[i] in 'hs' and t not in mo.unf[i])): continue
        ds = asg[(i, t)]
        if all((i, t, kind) and ((i, t, kind), d) in col for d in ds) and ds:
            for d in ds: x0[col[((i, t, kind), d)]] = 1; used.add(d)
        else: bad.append((i, t, kind))
    for b in bad:
        i, t, kind = b; r = M[i]['dataRequirements'][t]
        sub = dp_min_subset([c for c in cands[b] if c[1] not in used], r['lowerBound'], r['upperBound'], prefer_big=False)
        if sub is None: log(f'warm fail {b}'); continue
        for sz, d in sub: x0[col[(b, d)]] = 1; used.add(d)
    log(f'warm bad bins {len(bad)}')
    return x0
x0 = None if startf.endswith('.plan') else warm(json.load(open(startf)))
if x0 is not None:
    viol = [(q, mo.rows[q][3]) for q in range(mo.A.shape[0]) if not (mo.rlo[q] - 1e-6 <= mo.A.getrow(q).dot(x0)[0] <= mo.rhi[q] + 1e-6)]
    log(f'warm violations {len(viol)} {viol[:8]}')
st, x, bd = solve(set(), 120, x0)
if x is None:
    log(f'start infeasible {st}'); sys.exit()
obj = float(mo.cost @ np.round(x)); log(f'start obj {obj} {st} {time.time()-t0:.0f}s')
def slacks(x):
    return {t: int(round(mo.rhi[mo.massrow[t]] - mo.A.getrow(mo.massrow[t]).dot(np.round(x))[0])) for t in T}
log(f'slacks {slacks(x)}')
best = obj; it = 0
trained = lambda: [i for i in cur if cur[i] != '-']
ratio = sorted([i for i in range(len(M))], key=lambda i: -M[i]['value'] / max(1, M[i]['energyCost']))
while time.time() - t0 < tmax:
    it += 1
    tr = trained(); un = [i for i in ratio[:1500] if cur[i] == '-']
    free = set(rng.sample(tr, min(int(NT*SZ), len(tr)))) | set(rng.sample(un, min(int(NU*SZ), len(un))))
    if MODE == 'e1':
        FT_ = rng.choice(['in', 'ni', 'it', 'ic', 'nt', 'nc', 'tc', 'in'])
        free = set(E1) | {i for i in tr if cur[i] == 's'} | set(rng.sample(tr, min(int(NT*SZ), len(tr))))
    NB = int(NB0*SZ); ts = time.time()
    XFIX = np.round(x)
    st, xn, bd = solve(free, TL, x)
    if xn is None:
        log(f'it {it} none {st} {time.time()-t0:.0f}s'); continue
    on = float(mo.cost @ np.round(xn))
    log(f'it {it} {st} on {on} bd {bd:.1f} {time.time()-t0:.0f}s SZ {SZ:.2f} FT {FT_}')
    SZ = SZ * 1.15 if (st == 'Optimal' and time.time() - ts < TL / 3) else SZ * 0.85 if st != 'Optimal' else SZ
    if on >= obj - 1e-6:
        if on > obj + 0.5: log(f'it {it} {obj}->{on} bd {bd:.1f} {time.time()-t0:.0f}s slacks {slacks(xn)}')
        x, obj = xn, on
        p = exm.plan_from_x(mo, x)
        for i in cur: cur[i] = '-'
        for i in p['full']: cur[i] = 'f'
        for i in p['half']: cur[i] = 'h'
        for i in p['source']: cur[i] = 's'
        if on > best + 0.5 or it % 10 == 0:
            best = max(best, on); json.dump(p, open(f'planB_{TAG}.json', 'w'))
    if it % 10 == 0: log(f'it {it} obj {obj} {time.time()-t0:.0f}s')
log(f'END obj {obj}')
