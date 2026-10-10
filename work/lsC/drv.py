import sys, time, random, json
from lnsx import *
tmax = float(sys.argv[1]); seed = int(sys.argv[2]); tl = float(sys.argv[3]) if len(sys.argv) > 3 else 30
nR = int(sys.argv[4]) if len(sys.argv) > 4 else 20; nP = int(sys.argv[5]) if len(sys.argv) > 5 else 30
rng = random.Random(seed)
st = St(json.load(open(OUT))); log(f"start {st.val}")
t0 = time.time(); it = 0
nonF_ok = lambda st: [i for i in range(n) if i not in st.F and i not in st.src and i not in st.fixedT and i not in BAD]
while time.time() - t0 < tmax:
    it += 1
    t = rng.choice("ntic"); mode = rng.choice([0, 1, 2])
    Fs = [i for i in st.F if t in RQ[i]]
    L0 = RQ[rng.choice(Fs)][t][0]
    w = rng.choice([0.3, 0.6, 1.0])
    def dist(i): return abs(RQ[i][t][0] - L0) / (L0 + 2000) * (0.5 + rng.random())
    if mode == 0:  # same-type size neighbourhood
        R = sorted(Fs, key=dist)[:nR]
        cand = [i for i in nonF_ok(st) if t in RQ[i]]
        P = sorted(cand, key=lambda i: dist(i) + w * max(0, -RCF[i]) / 5)[:nP]
    elif mode == 1:  # rc guided: best non-F by rc, R by overlap
        cand = nonF_ok(st)
        P = sorted(cand, key=lambda i: max(0, -RCF[i]) * (0.5 + rng.random()))[:nP]
        tys = collections.Counter(tt for i in P for tt in RQ[i])
        R = sorted(st.F, key=lambda i: (RCF[i] + 0.2) * (0.5 + rng.random()) / (1 + sum(tys[tt] for tt in RQ[i])))[:nR]
    else:  # marginal F by rc + type-size neighbours
        R = sorted(st.F, key=lambda i: RCF[i] * (0.3 + rng.random()))[:nR // 2] + sorted(Fs, key=dist)[:nR // 2]
        R = list(dict.fromkeys(R))
        cand = [i for i in nonF_ok(st) if t in RQ[i]]
        P = sorted(cand, key=lambda i: max(0, -RCF[i]) * (0.5 + rng.random()) + dist(i))[:nP]
    ts = time.time()
    res, base, ob, N = solve_nb(st, R, P, tl)
    if res is None: log(f"it {it} m{mode} t{t} no sol"); continue
    gain = ob[0] - base
    log(f"it {it} m{mode} t{t} L0 {L0} N {N} base {base:.0f} obj {ob[0]:.0f} bnd {ob[1]:.1f} {time.time()-ts:.0f}s")
    if gain > 0.5:
        sol = apply(st, R, res)
        if sol is None: log("  apply failed"); continue
        sc = check(sol)
        log(f"  new score {sc}")
        if sc > st.val:
            sc2 = save_if_better(sol)
            json.dump(sol, open(f"best_C_{seed}.json", "w"))
            st = St(sol); log(f"  state now {st.val}")
