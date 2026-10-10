import json, sys, time, collections
sys.path.insert(0, '.')
from pack6 import pack_type
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
T1 = int(sys.argv[1]) if len(sys.argv)>1 else 9000
CT = int(sys.argv[2]) if len(sys.argv)>2 else 9000
tot_left=0
for t in "ntic":
    bins=[]
    for m in used:
        r = M[m]["dataRequirements"].get(t)
        if not r or (m,t) in covered: continue
        bins.append(((m,t), r["lowerBound"], r["upperBound"], m in full))
    items=[(d["size"], d["id"], not d["isCopyrighted"]) for d in D if d["dataType"]==t]
    t0=time.time()
    res, failed, left = pack_type(bins, items, T1, CT)
    lv = left.total()
    waste = sum(sum(D[i[1]]["size"] for i in its) - M[k[0]]["dataRequirements"][t]["lowerBound"] for k,its in res.items())
    print(t, "bins", len(bins), "failed", len(failed), "left items", len(left), "left vol", lv, "waste", waste, "time %.1f"%(time.time()-t0))
    tot_left += lv
print("total left", tot_left)

# detail for last type
import collections as C
agg=C.defaultdict(lambda:[0,0,0])
for k,its in res.items():
    L=M[k[0]]["dataRequirements"][t]["lowerBound"]
    s=sum(D[i[1]]["size"] for i in its)
    cls="L<=500" if L<=500 else "L<=9000" if L<=9000 else "L>9000"
    agg[(cls,len(its))][0]+=1; agg[(cls,len(its))][1]+=s-L; agg[(cls,len(its))][2]+=len(its)
for k,v in sorted(agg.items()): print(k,v)
print("failed", [(k, M[k[0]]["dataRequirements"][t]["lowerBound"]) for k in failed])
print("left", left.items)
