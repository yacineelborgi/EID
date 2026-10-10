import sys, time, random, json
from hyb import *
mode = sys.argv[1]; tl = float(sys.argv[2]); nP = int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
marg = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
st = St(json.load(open(OUT))); log(f"start {st.val}")
rng = random.Random(seed)
elig = [i for i in range(n) if i not in st.F and i not in st.src and i not in st.fixedT and i not in BAD]
if mode == 'global':
    R = sorted(st.F)
    P = sorted(elig, key=lambda i: -RCF[i])[:nP]
t0 = time.time()
res, base, ob, N = solve_h(st, R, P, tl, margin=marg, verbose=False)
log(f"N {N} base {base} obj {ob} {time.time()-t0:.0f}s")
if res and ob[0] > base + 0.5:
    newF = (st.F - set(R)) | {k[1] for k in res if k[0] == 'f'}
    log(f" out {sorted(st.F - newF)} in {sorted(newF - st.F)}")
    from real2 import realize2
    sol, failed = realize2(st, R, res, tmax=60)
    if sol is None: log(f" realize failed {failed}")
    else:
        sc = check(sol); log(f" realized score {sc}")
        json.dump(sol, open(f"hyb_{mode}_{seed}.json", "w"))
        if sc > 0: save_if_better(sol)
json.dump(sorted([list(k) for k in res]) if res else [], open(f"hres_{mode}_{seed}.json", "w"))
