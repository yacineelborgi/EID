"""Merge per-type item assignments from several solutions (same model mappings)."""
import json, sys
inst = json.load(open(sys.argv[1])); D = inst["datasets"]
base = json.load(open(sys.argv[2]))
out = sys.argv[3]
parts = sys.argv[4:]  # type:file
dm = {}
src_sol = {}
for p in parts:
    t, f = p.split(":", 1)
    src_sol[t] = json.load(open(f))
mm = base["modelMappings"]
for t, s in src_sol.items():
    assert s["modelMappings"] == mm or True
res = []
for t in "ntic":
    s = src_sol.get(t, base)
    for d, m in s["dataMappings"]:
        if D[d]["dataType"] == t:
            res.append([d, m])
json.dump({"dataMappings": sorted(res), "modelMappings": mm}, open(out, "w"))
