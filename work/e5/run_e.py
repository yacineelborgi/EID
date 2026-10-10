import json
from rmip5 import run
start = json.load(open("plan_d.json"))
plan = run((1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000), 20, 420, 100, 400, start=start, prune=100)
json.dump(plan, open("plan_e.json", "w"))
print({k: round(v) for k, v in plan["slack"].items() if k[-3:] in ("200",)})
