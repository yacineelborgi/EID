import json, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/big7")
import numpy as np
from data7 import load
d = load()
sol = json.load(open(sys.argv[1]))
ds, dt, dc, lb, has = d["ds"], d["dt"], d["dc"], d["lb"], d["has"]
srcs = {s for s, _ in sol["modelMappings"]}; tg = {t for _, t in sol["modelMappings"]}
dm = {}
for a, m in sol["dataMappings"]: dm.setdefault(m, []).append(a)
full = [m for m in dm if m not in srcs and m not in tg and all(dc[a] == 0 for a in dm[m])]
for t in range(4):
    fr = np.sort(ds[(dt == t) & (dc == 0)])
    l = np.sort([lb[m, t] for m in full if has[m, t]])
    k = len(l)
    wlb = int(np.maximum(0, fr[:k] - l).sum())
    act = sum(int(ds[[a for a in dm[m] if dt[a] == t]].sum()) - int(lb[m, t]) for m in full if has[m, t])
    # free used by non-full
    fo = sum(int(ds[a]) for m in dm if m not in full for a in dm[m] if dt[a] == t and dc[a] == 0)
    print("type", t, "full bins", k, "waste LB", wlb, "actual full waste", act, "free used by half/src", fo,
          "tiny(lb<=100)", int((l <= 100).sum()), "lb<=1000", int((l <= 1000).sum()))
