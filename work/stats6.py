import json, sys, collections, numpy as np
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
for t in "ntic":
  for pool in ("F","A"):
    L=[]
    for m in used:
        r = M[m]["dataRequirements"].get(t)
        if not r or (m,t) in covered: continue
        if (pool=="F") != (m in full): continue
        L.append(r["lowerBound"])
    L=np.array(L)
    its=np.array(sorted(d["size"] for d in D if d["dataType"]==t and (not d["isCopyrighted"] if pool=="F" else d["isCopyrighted"])))
    print(t,pool,"bins",len(L),"items",len(its),"vol",its.sum(),"sumL",L.sum(),"slack",its.sum()-L.sum(),
          "| L<=1000",(L<=1000).sum(),"1k-10k",((L>1000)&(L<=10000)).sum(),"10-20k",((L>10000)&(L<=20000)).sum(),"20-30k",((L>20000)&(L<=30000)).sum(),">30k",(L>30000).sum(), "maxL", L.max())
