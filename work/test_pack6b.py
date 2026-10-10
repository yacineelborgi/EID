import json, sys, time, collections
sys.path.insert(0, '.')
from pack6b import pack_type2
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
tlo=int(sys.argv[1]); thi=int(sys.argv[2])
tot=0
for t in "ntic":
    bins=[]
    for m in used:
        r = M[m]["dataRequirements"].get(t)
        if not r or (m,t) in covered: continue
        bins.append(((m,t), r["lowerBound"], r["upperBound"], m in full))
    items=[(d["size"], d["id"], not d["isCopyrighted"]) for d in D if d["dataType"]==t]
    t0=time.time()
    res, failed, left = pack_type2(bins, items, tlo=tlo, thi=thi)
    lv=sum(i[0] for i in left)
    fv=sum(M[k[0]]["dataRequirements"][t]["lowerBound"] for k in failed)
    print(t,"failed",len(failed),"failedL",fv,"left",len(left),lv,"time %.1f"%(time.time()-t0))
    tot+=lv-fv
print("net", tot)
