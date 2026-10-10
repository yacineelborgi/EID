"""MIP-LNS on e5 relaxation around a plan (per-type margins). Each iteration frees a neighbourhood of models
(LP-dev guided + random), fixes the rest to the incumbent modes, solves with cutoff incumbent+1.
usage: lns5b.py plan.json out.json mfs mts nfree tl iters seed"""
import sys, json, time, os, random, pickle
import numpy as np, highspy
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/e5")
from relax import build, to_highs, INF
from rmip5 import set_mode, plan_mode
plan = json.load(open(sys.argv[1])); out = sys.argv[2]
MFS = [int(x) for x in sys.argv[3].split(',')]; MTS = [int(x) for x in sys.argv[4].split(',')]
NF = int(sys.argv[5]); TL = float(sys.argv[6]); IT = int(sys.argv[7]); rnd = random.Random(int(sys.argv[8]))
taus = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)
inst = json.load(open("/home/user/EID/datasets/5_energy.json")); M = inst["models"]; nM = len(M)
mdl = build(inst, taus)
for r, nm in enumerate(mdl.rnames):
    for k, t in enumerate("ntic"):
        if nm.startswith("free" + t): mdl.rhi[r] -= MFS[k]
        if nm.startswith("tot" + t): mdl.rhi[r] -= MTS[k]
LP = pickle.load(open('lp5_100_400.pkl', 'rb')); modes = LP['modes']
sec = {i: sorted(float(v) for v in d.values())[1] if len(d) > 1 else 1e9 for i, d in modes.items()}
def xvec(pm):
    xs = np.zeros(len(mdl.c))
    for i, md in pm.items():
        if md == "full" and i in mdl.fv: xs[mdl.fv[i]] = 1
        elif md == "src" and i in mdl.sv: xs[mdl.sv[i]] = 1
        elif md.startswith("half") and i in mdl.hv:
            xs[mdl.hv[i]] = 1
            for t in md[4:]: xs[mdl.zv[(i, t)]] = 1
    return xs
def val(p): return sum(M[i]['value'] for i in p['full']) + sum(M[i]['value'] // 2 for i in p['half'])
best = val(plan); print('start', best, flush=True)
h, _ = to_highs(mdl, integer=True)
h.setOptionValue("mip_rel_gap", 0.0); h.setOptionValue("mip_abs_gap", 0.5); h.setOptionValue("threads", 1)
t0 = time.time()
cand_dev = sorted(range(nM), key=lambda i: sec[i])
for it in range(IT):
    pm = plan_mode(plan)
    # neighbourhood: models whose current mode has positive dev, low second-best dev, + random
    pos = [i for i in range(nM) if float(modes[i].get(pm.get(i, 'off'), 0)) > 1e-9]
    k1 = rnd.randint(NF // 3, NF // 2)
    free = set(rnd.sample(pos, min(len(pos), NF // 4))) | set(rnd.sample(cand_dev[:3 * NF], k1)) | set(rnd.sample(range(nM), NF // 6))
    lo, up = list(mdl.lo), list(mdl.up)
    for i in range(nM):
        if i in free: continue
        set_mode(mdl, lo, up, i, pm.get(i, 'off'))
    h.changeColsBounds(len(lo), np.arange(len(lo), dtype=np.int32), np.array(lo, float), np.array(up, float))
    h.setOptionValue("time_limit", TL); h.setOptionValue("random_seed", it)
    h.setOptionValue("objective_bound", best + 0.5)  # maximize: prune nodes whose bound <= best? (cutoff)
    xs = xvec(pm); hs = highspy.HighsSolution(); hs.col_value = list(xs); hs.value_valid = True; h.setSolution(hs)
    h.run(); info = h.getInfo()
    x = np.array(h.getSolution().col_value)
    obj = info.objective_function_value if info.primal_solution_status == 2 else -1
    print(f'it {it} free {len(free)} obj {obj:.0f} bound {info.mip_dual_bound:.1f} {h.modelStatusToString(h.getModelStatus())} {time.time()-t0:.0f}s', flush=True)
    if obj > best + 0.5:
        p = {"full": [], "half": [], "source": [], "srcreq": []}
        for i, j in mdl.fv.items():
            if x[j] > 0.5: p["full"].append(i)
        for i, j in mdl.hv.items():
            if x[j] > 0.5: p["half"].append(i)
        for i, j in mdl.sv.items():
            if x[j] > 0.5: p["source"].append(i)
        for (i, t), j in mdl.zv.items():
            if x[j] > 0.5: p["srcreq"].append([i, t])
        p['obj'] = val(p)
        if p['obj'] > best:
            best = p['obj']; plan = p; json.dump(p, open(out, 'w')); print('  NEW', best, flush=True)
