import json, sys, time, pickle
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6, highspy
from enc import encode, encode_plan
inst = json.load(open('/home/user/EID/datasets/6_shortage.json'))
inc_path = sys.argv[1]; tl = float(sys.argv[2]); out = sys.argv[3]
inc_val = float(sys.argv[4]) if len(sys.argv) > 4 else None
margs = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
marg = {(k[0], k[1]): v for k, v in margs.items()}
t0 = time.time()
mo = mip6.build(inst, marg=marg)
print("built", mo.A.shape, mo.A.nnz, round(time.time() - t0), flush=True)
sol = json.load(open(inc_path))
x0 = encode_plan(mo, inst, sol) if "full" in sol else encode(mo, inst, sol)
# check incumbent feasibility
Ax = mo.A @ x0
viol = np.maximum(mo.rlo - Ax, 0) + np.maximum(Ax - mo.rhi, 0)
print("incumbent obj", mo.cost @ x0, "max viol", viol.max(), mo.names[int(viol.argmax())], flush=True)
h = mip6.to_highs(mo, integer=False)
h.run()
lpv = h.getInfo().objective_function_value
s = h.getSolution()
y = np.array(s.row_dual); xl = np.array(s.col_value)
rc = mo.cost - mo.A.T @ y
print("LP", lpv, round(time.time() - t0), flush=True)
np.savez(out + "_lp.npz", y=y, x=xl, rc=rc)
if inc_val is None:
    inc_val = mo.cost @ x0
gap = lpv - inc_val
# fix binaries at 0 with rc < -gap (max problem: increasing var costs -rc)
fix0 = [j for j in range(mo.nV) if xl[j] < 1e-9 and rc[j] < -(gap + 1.0) and mo.ub[j] > 0]
fix1 = [j for j in range(mo.nV) if xl[j] > 1 - 1e-9 and rc[j] > (gap + 1.0) and mo.lb[j] < 1]
print("fix0", len(fix0), "fix1", len(fix1), "of", mo.nV, flush=True)
viol0 = [j for j in fix0 if x0[j] > 0.5]; viol1 = [j for j in fix1 if x0[j] < 0.5]
print("incumbent conflicts", len(viol0), len(viol1), flush=True)
ub = mo.ub.copy(); lb = mo.lb.copy()
ub[fix0] = 0; lb[fix1] = 1
mo.ub = ub; mo.lb = lb
h = mip6.to_highs(mo, integer=True, verbose=True)
h.setOptionValue("time_limit", tl)
h.setOptionValue("mip_rel_gap", 1e-6)
h.setOptionValue("log_file", out + "_highs.log")
if not viol0 and not viol1:
    st = h.setSolution(highspy.HighsSolution() if False else None) if False else None
    sol_ = highspy.HighsSolution()
    sol_.col_value = list(x0)
    sol_.value_valid = True
    print("setSolution", h.setSolution(sol_), flush=True)
h.run()
info = h.getInfo()
x = np.array(h.getSolution().col_value)
plan = mip6.plan_from_x(mo, x)
plan["obj"] = info.objective_function_value
plan["bound"] = info.mip_dual_bound
plan["lp"] = lpv
print("MIP obj", plan["obj"], "bound", plan["bound"], "time", round(time.time() - t0), flush=True)
json.dump(plan, open(out + "_plan.json", "w"))
