"""Solve the selection MIP with the tiny-bin correction and cache the plan.
usage: python plan_med.py datasets/X.json time alpha tiny_lb margin [gap]"""
import json, os, sys, time
from select_mip import build_and_solve, load

path, tl, alpha, tiny, margin = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], float(sys.argv[5])
tiny_arg = dict(zip("ntic", map(int, tiny.split(",")))) if "," in tiny else int(tiny)
gap = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-7
inst = load(path)
t0 = time.time()
plan = build_and_solve(inst, alpha_free=alpha, alpha_tot=alpha, time_limit=tl, tiny_lb=tiny_arg,
                       free_margin=margin, gap=gap, verbose=True)
os.makedirs("plans", exist_ok=True)
pf = f"plans/{os.path.basename(path)[:-5]}_tiny{tiny.replace(',', '-')}_m{margin:g}_a{alpha}.json"
json.dump(plan, open(pf, "w"))
print("DONE", pf, "obj", plan["obj"], "bound", plan["bound"], "full", len(plan["full"]), "half", len(plan["half"]),
      "src", len(plan["source"]), "time", round(time.time() - t0), flush=True)
