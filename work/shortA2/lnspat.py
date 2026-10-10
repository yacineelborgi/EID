"""pattern-LNS on SA-format state. usage: lnspat.py in.txt start.out out.out seconds seed K [tl]"""
import sys, time, random
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA2')
from patmip import build, solve
inf, stf, outf, secs, seed, K = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
tl = float(sys.argv[7]) if len(sys.argv) > 7 else 20
v = open(inf).read().split("\n"); nb, ni = map(int, v[0].split())
B = []
for x in v[1:1 + nb]:
    L, U, F, W = x.split(); B.append((0, int(L), int(U), bool(int(F))))
I = []
for x in v[1 + nb:1 + nb + ni]:
    s, f = x.split(); I.append((int(s), 0, bool(int(f))))
a = [int(x) for x in open(stf).read().split()]
rng = random.Random(seed)
def state():
    S = [0] * nb; per = {}
    for j, b in enumerate(a):
        if b >= 0: S[b] += I[j][0]; per.setdefault(b, []).append(j)
    return S, per
def score():
    S, per = state()
    fails = [b for b in range(nb) if S[b] < B[b][1]]
    w = sum(S[b] - B[b][1] for b in range(nb) if S[b] >= B[b][1])
    return len(fails), sum(B[b][1] - S[b] for b in fails), w, fails
print("start", score()[:3], flush=True)
t0 = time.time(); it = 0
while time.time() - t0 < secs:
    it += 1
    S, per = state()
    fails = [b for b in range(nb) if S[b] < B[b][1]]
    filled = [b for b in range(nb) if S[b] >= B[b][1] and B[b][1] > 1]
    wasty = [b for b in filled if S[b] - B[b][1] > 0 and len(per.get(b, [])) >= 2]
    k1 = min(len(wasty), int(K * rng.choice([0.3, 0.5, 0.7])))
    sub = set(fails) | set(rng.sample(wasty, k1))
    others = [b for b in filled if b not in sub]
    sub |= set(rng.sample(others, min(len(others), K - len(sub))))
    sub = sorted(sub)
    its = [j for j in range(ni) if a[j] < 0] + [j for b in sub for j in per.get(b, [])]
    lb = [B[b] for b in sub]; li = [I[j] for j in its]
    pos = {j: q for q, j in enumerate(its)}
    start = {q: [pos[j] for j in per[b]] for q, b in enumerate(sub) if b in per and S[b] >= B[b][1]}
    W2 = rng.choice([2, 5, 10]); 
    pats = build(lb, li, W2=W2, W1small=3000, W1big=1500, K3=200, K4=100, seed=rng.randrange(10**6), extra=list(start.items()))
    pen = [50000.0] * len(sub)
    cur = sum(S[b] - B[b][1] for b in sub if S[b] >= B[b][1]) + 50000.0 * sum(1 for b in sub if S[b] < B[b][1])
    r = solve(lb, li, pats, pen, tl, start=start, log=False)
    if r is None: continue
    asg, obj, bnd = r
    if obj < cur - 0.5:
        for b in sub:
            for j in per.get(b, []): a[j] = -1
        for q, js in asg.items():
            for jj in js: a[its[jj]] = sub[q]
        nf, d, w, fl = score()
        print(f"it {it} t={time.time()-t0:.0f}s K {len(sub)} pats {len(pats)} W2 {W2} cur {cur:.0f} -> {obj:.0f} | fails {nf} def {d} waste {w}", flush=True)
        open(outf, "w").write("\n".join(map(str, a)) + "\n")
nf, d, w, fl = score()
print("end", nf, d, w, [B[b][1] for b in fl], flush=True)
open(outf, "w").write("\n".join(map(str, a)) + "\n")
