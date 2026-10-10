"""Solve the mip6 relaxation model as a MIP on any instance, save the plan."""
import json, sys, time
import numpy as np
sys.path.insert(0, '.')
import mip6
inst = json.load(open(sys.argv[1]))
tl = float(sys.argv[2]); out = sys.argv[3]
mo = mip6.build(inst)
h = mip6.to_highs(mo, integer=True, threads=1, verbose=False)
h.setOptionValue("time_limit", tl)
h.setOptionValue("mip_rel_gap", 1e-9)
t0 = time.time()
h.run()
info = h.getInfo()
x = np.array(h.getSolution().col_value)
plan = mip6.plan_from_x(mo, x)
plan["obj"] = info.objective_function_value
plan["bound"] = info.mip_dual_bound
json.dump(plan, open(out, "w"))
print("obj", info.objective_function_value, "bound", info.mip_dual_bound, "time", time.time() - t0,
      "full", len(plan["full"]), "half", len(plan["half"]), "src", len(plan["source"]))
