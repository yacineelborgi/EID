import json, sys, numpy as np, collections
inst = json.load(open("../datasets/6_shortage.json"))
M = inst["models"]; D = inst["datasets"]
sol = json.load(open(sys.argv[1] if len(sys.argv)>1 else "../solutions/6_shortage_submission.json"))
srcs = {a for a,b in sol["modelMappings"]}
tg = collections.defaultdict(list)
for a,b in sol["modelMappings"]: tg[b].append(a)
covered = set()
for a,b in sol["modelMappings"]:
    covered.add((b, next(iter(M[a]["dataRequirements"]))))
fill = collections.defaultdict(list)
for db,m in sol["dataMappings"]: fill[(m, D[db]["dataType"])].append(db)
used = set(m for _,m in sol["dataMappings"]) | srcs | set(tg)
cop = collections.defaultdict(int)
for db,m in sol["dataMappings"]: cop[m] |= D[db]["isCopyrighted"]
full = {m for m in used if m not in srcs and not cop[m] and not tg[m]}
for t in "ntic":
    fs = sorted(d["size"] for d in D if d["dataType"]==t and not d["isCopyrighted"])
    al = sorted(d["size"] for d in D if d["dataType"]==t)
    binsF=[]; binsA=[]; wF=0; wA=0; volF=0; volA=0
    for m in used:
        r = M[m]["dataRequirements"].get(t)
        if not r or (m,t) in covered: continue
        L = r["lowerBound"]; s = sum(D[x]["size"] for x in fill[(m,t)])
        binsA.append(L); volA += s
        if m in full: binsF.append(L); volF += s
    def lpcons(bins, items):
        bins=sorted(bins); pre=np.concatenate([[0],np.cumsum(items)])
        best=sum(bins)
        for th in [1,30,100,200,300,450,600,800,1000,1300,1700,2200,2800,3500,4500,6000,8000,10000]:
            k=sum(1 for b in bins if b<=th)
            v=sum(b for b in bins if b>th)+pre[min(k,len(items))]
            best=max(best,v)
        # exact: k smallest bins -> k smallest items
        return best
    cF=lpcons(binsF, fs); cA=lpcons(binsA, al)
    print(t, "FREE: nbins",len(binsF),"sumL",sum(binsF),"LPcons",cF,"actual",volF,"cap",sum(fs),"| ALL: nbins",len(binsA),"sumL",sum(binsA),"LPcons",cA,"actual",volA,"cap",sum(al), " slackA", sum(al)-cA)
