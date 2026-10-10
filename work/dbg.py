import json, sys, collections
sys.path.insert(0, '.')
from pack6b import make_bases, closer_match
inst = json.load(open("../datasets/6_shortage.json"))
M = inst["models"]; D = inst["datasets"]
sol = json.load(open("../solutions/6_shortage_submission.json"))
srcs = {a for a,b in sol["modelMappings"]}
tg = collections.defaultdict(list)
for a,b in sol["modelMappings"]: tg[b].append(a)
covered=set((b, next(iter(M[a]["dataRequirements"]))) for a,b in sol["modelMappings"])
used = set(m for _,m in sol["dataMappings"]) | srcs | set(tg)
cop = collections.defaultdict(int)
for db,m in sol["dataMappings"]: cop[m] |= D[db]["isCopyrighted"]
full = {m for m in used if m not in srcs and not cop[m] and not tg[m]}
t="n"
bins=[((m,t), M[m]["dataRequirements"][t]["lowerBound"], M[m]["dataRequirements"][t]["upperBound"]) for m in used if t in M[m]["dataRequirements"] and (m,t) not in covered and m in full]
items=[(d["size"], d["id"], True) for d in D if d["dataType"]==t and not d["isCopyrighted"]]
singles=[b for b in bins if b[1]<=10000]; multis=[b for b in bins if b[1]>10000]
bases, rest = make_bases(multis, items, 300, 9500)
print(len(singles), len(multis), len(items), len(rest))
ds=[d for k,(b,d) in bases.items()]
import numpy as np
print("resid pct", np.percentile(ds,[0,10,50,90,100]))
print("nbase", collections.Counter(len(b) for b,d in bases.values()))
reqs=[(k,lb,ub-lb) for k,lb,ub in singles]+[(k,d,[b for b in bins if b[0]==k][0][2]-[b for b in bins if b[0]==k][0][1]) for k,(b,d) in bases.items()]
res,un=closer_match(reqs,rest)
print("matched",len(res),"un",len(un))
print([r for r in reqs if r[0] in set(un)][:10])
print(sorted(i[0] for i in rest)[-20:])
