"""margin MIP with rc-fixing at target T. usage: run_m.py mA mF T tl out [start.json] [seed]"""
import json, sys, time, os
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortB")
import mip6, highspy
from enc import encode_plan, encode
inst = json.load(open('/home/user/EID/datasets/6_shortage.json'))
MG = json.loads(sys.argv[1]); mA = 0; mF = 0; T = float(sys.argv[3]); tl = float(sys.argv[4]); out = sys.argv[5]
start = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != '-' else None
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
marg = {}
for t in 'ntic':
    marg[('A', t)] = MG[t][0]; marg[('F', t)] = MG[t][1]
t0 = time.time()
mo = mip6.build(inst, marg=marg)
h = mip6.to_highs(mo, integer=False)
h.run()
lpv = h.getInfo().objective_function_value
s = h.getSolution(); y = np.array(s.row_dual); xl = np.array(s.col_value)
rc = mo.cost - mo.A.T @ y
if T < 0:
    T = lpv + T
print("LP", lpv, "T", T, round(time.time() - t0), flush=True)
gap = lpv - T
fix0 = [j for j in range(mo.nV) if xl[j] < 1e-9 and rc[j] < -(gap + 0.5) and mo.ub[j] > 0]
fix1 = [j for j in range(mo.nV) if xl[j] > 1 - 1e-9 and rc[j] > (gap + 0.5) and mo.lb[j] < 1]
print("gap", gap, "fix0", len(fix0), "fix1", len(fix1), "of", mo.nV, flush=True)
x0 = None
if start:
    sol = json.load(open(start))
    x0 = encode_plan(mo, inst, sol) if "full" in sol else encode(mo, inst, sol)
    Ax = mo.A @ x0
    viol = np.maximum(mo.rlo - Ax, 0) + np.maximum(Ax - mo.rhi, 0)
    print("start obj", mo.cost @ x0, "maxviol", viol.max(), mo.names[int(viol.argmax())], flush=True)
    c0 = sum(1 for j in fix0 if x0[j] > 0.5); c1 = sum(1 for j in fix1 if x0[j] < 0.5)
    print("start conflicts", c0, c1, flush=True)
mo.ub = mo.ub.copy(); mo.lb = mo.lb.copy()
mo.ub[fix0] = 0; mo.lb[fix1] = 1
h = mip6.to_highs(mo, integer=True, verbose=True)
h.setOptionValue("time_limit", tl)
h.setOptionValue("mip_rel_gap", 1e-7)
h.setOptionValue("random_seed", seed)
h.setOptionValue("log_file", out + "_highs.log")
if x0 is not None:
    so = highspy.HighsSolution(); so.col_value = list(x0); so.value_valid = True
    print("setSolution", h.setSolution(so), flush=True)
h.run()
info = h.getInfo()
x = np.array(h.getSolution().col_value)
if len(x) == mo.nTot:
    plan = mip6.plan_from_x(mo, x)
    plan["obj"] = info.objective_function_value; plan["bound"] = info.mip_dual_bound; plan["lp"] = lpv
    plan["marg"] = MG
    json.dump(plan, open(out + "_plan.json", "w"))
    print("MIP obj", plan["obj"], "bound", plan["bound"], "time", round(time.time() - t0), flush=True)
else:
    print("no solution", h.modelStatusToString(h.getModelStatus()), flush=True)
