s=open('lnsR.py').read()
s=s.replace("""x0 = None if startf.endswith('.plan') else warm(json.load(open(startf)))""","""def warm_plan():
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict
    sys.path.insert(0, '/home/user/EID'); from exact_pack import dp_min_subset
    x0 = np.zeros(nC)
    for i, st_ in cur.items():
        d = {'f': mo.fv, 'h': mo.hv, 's': mo.sv}.get(st_)
        if d is not None: x0[d[i]] = 1
    cands = defaultdict(list)
    for k, b, did, sz in mo.xinfo: cands[b].append((k, did, sz))
    for t in T:
        bins = [b for b in mo.sbins if b[1] == t and ((b[2] == 'F' and cur[b[0]] == 'f') or (b[2] == 'H' and cur[b[0]] in 'hs' and t not in mo.unf[b[0]]))]
        items = sorted({did for b in bins for k, did, sz in cands[b]})
        ix = {d: q for q, d in enumerate(items)}
        C = np.full((len(bins), len(items)), 1e9)
        for p_, b in enumerate(bins):
            r = M[b[0]]['dataRequirements'][t]
            for k, did, sz in cands[b]:
                if sz >= r['lowerBound']: C[p_, ix[did]] = sz * (0 if exm.D[did]['isCopyrighted'] else 1) + 1e-3 * sz
        rr, cc = linear_sum_assignment(C)
        used = set(); left = []
        colof = {(b, did): k for b in bins for k, did, sz in cands[b]}
        for p_, q in zip(rr, cc):
            if C[p_, q] < 1e8: x0[colof[(bins[p_], items[q])]] = 1; used.add(items[q])
            else: left.append(bins[p_])
        for b in left:
            r = M[b[0]]['dataRequirements'][t]
            sub = dp_min_subset([(sz, did) for k, did, sz in cands[b] if did not in used], r['lowerBound'], r['upperBound'], prefer_big=False)
            if sub is None: log(f'warm fail {b}'); continue
            for sz, did in sub: x0[colof[(b, did)]] = 1; used.add(did)
    for t in T:
        q = mo.massrow[t]; v = mo.A.getrow(q).dot(x0)[0]
        x0[VC[t]] = max(0.0, v + x0[VC[t]] - mo.rhi[q])
    return x0
x0 = warm_plan() if startf.endswith('.plan') else warm(json.load(open(startf)))""")
s=s.replace("""st, x, bd = solve(set(), 120, x0)""","""log(f'warm viol {[round(x0[VC[t]]) for t in T]}')
st, x, bd = solve(set(), float(os.environ.get('T0', '60')), x0)""")
open('lnsR.py','w').write(s)
