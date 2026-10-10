"""Recreate step in one HiGHS process: residual addition MIP (resid_mip) then role relabeling MIP.
usage: python recreate.py inst sol_in params.json sol_out"""
import json, sys
sys.path.insert(0, '.')
from sol6 import State
import resid_mip, roles

inst = json.load(open(sys.argv[1]))
st = State(inst, json.load(open(sys.argv[2])))
p = json.load(open(sys.argv[3]))
hint = {int(k): (v[0], v[1]) for k, v in p.get("hint", {}).items()}
g, b = resid_mip.solve(st, p.get("multi_cands", []), p.get("extra_mono", []), hint, p.get("tlim", 60),
                       p.get("cap_src", 120), p.get("cap_data", 120), p.get("emax_src", 3e11), p.get("max_tgt", 300))
print("resid", g, b, flush=True)
try:
    r = roles.solve_roles(st, tlim=p.get("roles_tlim", 120))
    if r["obj"] >= r["cur"]:
        roles.apply_roles(st, r)
    print("roles", r["obj"], r["cur"], flush=True)
except AssertionError as e:
    print("roles failed", e, flush=True)
json.dump(st.to_sol(), open(sys.argv[4], "w"))
