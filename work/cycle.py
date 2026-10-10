"""One improvement cycle: compact each type in parallel (subprocesses) -> merge ->
residual add -> role MIPs. Saves to solutions/ when better.
usage: python cycle.py inst sol_in out rounds pairs seed"""
import json, sys, os, subprocess, tempfile
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sol6 import State
import resid_add, roles, roles2
from solve import strict_ok

inst_p, sol_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
rounds, pairs, seed = sys.argv[4], sys.argv[5], int(sys.argv[6])
tmp = tempfile.mkdtemp(dir=os.path.dirname(out_p) or ".")
procs = []
for t in "ntic":
    procs.append(subprocess.Popen([sys.executable, "compact.py", inst_p, sol_p, t, rounds, pairs, str(seed), f"{tmp}/c_{t}.json"],
                                  stdout=open(f"{tmp}/c_{t}.log", "w"), stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
subprocess.run([sys.executable, "merge_types.py", inst_p, sol_p, f"{tmp}/merged.json"] + [f"{t}:{tmp}/c_{t}.json" for t in "ntic"], check=True)
inst = json.load(open(inst_p))
st = State(inst, json.load(open(f"{tmp}/merged.json")))
print("after compaction", st.score(), flush=True)
for t in "ntic":
    print(t, "pool", sum(d["size"] for d in st.D if d["dataType"] == t and d["id"] not in st.item_bin))
g = resid_add.add_models(st, tlim=120)
print("after resid", st.score(), flush=True)
r = roles.solve_roles(st, tlim=240)
roles.apply_roles(st, r)
print("after roles", st.score(), flush=True)
r = roles2.solve_roles2(st, tlim=600)
if r["obj"] >= r["cur"]:
    roles2.apply_roles2(st, r)
print("after roles2", st.score(), flush=True)
sc, ok, msg = st.score()
sol = st.to_sol()
if ok and strict_ok(inst, sol):
    json.dump(sol, open(out_p, "w"))
    print("wrote", out_p, sc)
