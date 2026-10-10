"""realize a hybrid MIP result with exact packing + MIP repair."""
import sys, json, random, collections, time
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA")
from ptype import mip_assign
from hyb import *

NEWY=set()
def pack_t(st, t, fixed_dm, newbins, seed=0, tmax=60, log=log):
    """fixed_dm: d->key for items fixed (not touchable except as donors listed in donorbins).
    newbins: list of (key, lb, ub). Returns fill dict key->[d] or None."""
    rng = random.Random(seed)
    freeT = [d['id'] for d in D if d['dataType'] == t and not d['isCopyrighted']]
    pool = [(D[d]['size'], d) for d in freeT if d not in fixed_dm]
    best = None
    for sd, order in enumerate(["lb_desc", "lb_asc", "random", "width", "random", "random"]):
        fill, still, left = pack_type(newbins, pool, seed=seed + sd, order=order)
        if best is None or len(still) < len(best[1]): best = (fill, still)
        if not still: break
    fill, still = best
    asg = {k: [d for s, d in v[2]] for k, v in fill.items()}
    binfo = {k: (lb, ub) for k, lb, ub in newbins}
    # donors: all fixed F bins of type t + packed new bins
    fixb = collections.defaultdict(list)
    for d, k in fixed_dm.items(): fixb[k].append(d)
    for k, ds in fixb.items():
        i = k[0]; binfo.setdefault(k, RQ[i][t])
    asg.update({k: v for k, v in fixb.items() if k not in asg})
    failed = list(still)
    t0 = time.time(); it = 0
    while failed and time.time() - t0 < tmax:
        it += 1
        b0 = failed[0]
        used = {d for v in asg.values() for d in v}
        pl = [d for d in freeT if d not in used]
        L0 = binfo[b0][0]
        keys = list(asg)
        nd = rng.choice([8, 15, 25, 40])
        if it % 2: donors = sorted(keys, key=lambda k: abs(binfo[k][0] - L0) * (0.5 + rng.random()))[:nd]
        else: donors = rng.sample(keys, min(nd, len(keys)))
        bl = [b0] + donors
        its = pl + [d for k in donors for d in asg[k]]
        if it % 3 == 0:
            items = [(D[d]['size'], d, True) for d in its]
            bins = [(k, binfo[k][0], binfo[k][1], True) for k in bl]
            r = mip_assign(bins, items, tl=10)
            if r is None: continue
            _, out, filled, _, _ = r
            for q, k in enumerate(bl): asg[k] = [its[j] for j in out[q]]
        else:
            av = set(its); newf = {}; good = True
            for k in sorted(bl, key=lambda k: -binfo[k][0] * (0.8 + 0.4 * rng.random())):
                rr = dp_min_subset([(D[d]['size'], d) for d in av], binfo[k][0], binfo[k][1])
                if rr is None: good = False; break
                newf[k] = [d for _, d in rr]; av.difference_update(newf[k])
            if not good: continue
            asg.update(newf)
        failed.pop(0)
        log(f"   repaired {b0} type {t} it {it} nd {nd}")
    used = {d for v in asg.values() for d in v}
    ws = sorted(((sum(D[d]['size'] for d in v) - binfo[k][0]), binfo[k][0], len(v)) for k, v in asg.items() if k not in fixb or k[0] in NEWY)
    log(f"  waste>0 (non-fixed) {[w for w in ws if w[0] > 0][-25:]}")
    log(f"  type {t} leftover {sum(D[d]['size'] for d in freeT if d not in used)} failed {len(failed)}")
    if failed: return None, failed
    return asg, []

def realize2(st, R, res, seed=0, tmax=60, log=log):
    R = set(R)
    newF = (st.F - R) | {k[1] for k in res if k[0] == 'f'}
    newH = {k[1] for k in res if k[0] == 'h'} | st.fixedT
    dm = {}
    for s in st.src:
        for d in st.used[s]: dm[d] = s
    ysm = {(k[2], D[k[1]]['dataType']): [k[1]] for k in res if k[0] == 'y'}
    ysm.update({(k[1], k[2]): list(k[3]) for k in res if k[0] == 'p'})
    for t in "ntic":
        fixed = {}
        for i in st.F - R:
            for d in st.used[i]:
                if D[d]['dataType'] == t: fixed[d] = (i, t)
        for (i, tt), ds in ysm.items():
            if tt == t:
                for d in ds: fixed[d] = (i, t)
        newbins = [((i, t), RQ[i][t][0], RQ[i][t][1]) for i in newF - (st.F - R) if t in RQ[i] and (i, t) not in ysm]
        asg, failed = pack_t(st, t, fixed, newbins, seed=seed, tmax=tmax, log=log)
        if asg is None: return None, failed
        for k, ds in asg.items():
            for d in ds: dm[d] = k[0]
    leftover = {t: set() for t in "ntic"}
    for d in D:
        if d['id'] not in dm: leftover[d['dataType']].add(d['id'])
    keep = {}
    for i in newH:
        if i in st.H and all(d in leftover[D[d]['dataType']] for d in st.used[i]):
            keep[i] = st.used[i]
            for d in st.used[i]: leftover[D[d]['dataType']].discard(d)
    for i in sorted(newH - set(keep), key=lambda i: -max(r[0] for r in RQ[i].values())):
        ds = []
        for t, (lb, ub) in RQ[i].items():
            if any(next(iter(M[s]['dataRequirements'])) == t for s in st.tg.get(i, [])): continue
            items = [(D[d]['size'], d) for d in leftover[t] if D[d]['isCopyrighted']]
            rr = dp_min_subset(items, lb, ub)
            if rr is None: return None, [('half', i, t)]
            for _, d in rr: leftover[t].discard(d); ds.append(d)
        keep[i] = ds
    for i, ds in keep.items():
        for d in ds: dm[d] = i
    return {"dataMappings": [[d, m] for d, m in dm.items()], "modelMappings": [list(x) for x in st.sol['modelMappings']]}, []

if __name__ == "__main__":
    st = St(json.load(open(OUT)))
    res = {tuple(tuple(x) if isinstance(x, list) else x for x in k) for k in json.load(open(sys.argv[1]))}
    R = sorted(st.F)
    sol, failed = realize2(st, R, res, tmax=float(sys.argv[2]) if len(sys.argv) > 2 else 120)
    if sol is None: log(f"FAILED {failed}")
    else:
        sc = check(sol); log(f"realized {sc}")
        json.dump(sol, open(sys.argv[1] + ".sol.json", "w"))
        if sc > 0: save_if_better(sol)
