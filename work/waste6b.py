import json, sys, numpy as np, collections
inst = json.load(open("../datasets/6_shortage.json"))
M = inst["models"]; D = inst["datasets"]
sol = json.load(open(sys.argv[1] if len(sys.argv)>1 else "../solutions/6_shortage_submission.json"))
srcs = {a for a,b in sol["modelMappings"]}
covered=set((b, next(iter(M[a]["dataRequirements"]))) for a,b in sol["modelMappings"])
fill = collections.defaultdict(list)
for db,m in sol["dataMappings"]: fill[(m, D[db]["dataType"])].append(db)
agg=collections.defaultdict(lambda:[0,0])
for (m,t),ds in fill.items():
    if (m,t) in covered: 
        agg["wasted-on-covered"][0]+=1; agg["wasted-on-covered"][1]+=sum(D[x]["size"] for x in ds); continue
    L=M[m]["dataRequirements"][t]["lowerBound"]; s=sum(D[x]["size"] for x in ds)
    cls = "L<=500" if L<=500 else "L<=2000" if L<=2000 else "L<=10000" if L<=10000 else "L>10000"
    key=(cls, min(len(ds),4))
    agg[key][0]+=1; agg[key][1]+=s-L
for k,v in sorted(agg.items(), key=str): print(k, "nbins",v[0],"waste",v[1], "avg", v[1]/max(v[0],1))
