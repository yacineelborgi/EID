import json, sys, collections
sys.path.insert(0,'.')
from pack6b import closer_match
from sol6 import State
inst=json.load(open("../datasets/6_shortage.json")); sol=json.load(open(sys.argv[1] if len(sys.argv)>1 else "../solutions/6_shortage_submission.json"))
st=State(inst,sol); M,D=st.M,st.D
full={m for m in st.used if m not in st.src and st.is_full(m)}
cov=st.covered()
for t in "ntic":
    bins=[]
    for m in st.used:
        r=M[m]["dataRequirements"].get(t)
        if not r or (m,t) in cov: continue
        bins.append(((m,t),r["lowerBound"],r["upperBound"], m in full))
    cur_w = {b[0]: sum(D[i]["size"] for i in st.bin_items[b[0]]) - b[1] for b in bins}
    sing=[b for b in bins if b[1]<=10000]
    free=[(d["size"],d["id"],True) for d in D if d["dataType"]==t and not d["isCopyrighted"]]
    allit=[(d["size"],d["id"],not d["isCopyrighted"]) for d in D if d["dataType"]==t]
    # F singles on free items
    sF=[(k,lb,ub-lb) for k,lb,ub,fo in sing if fo]
    rF,uF=closer_match(sF,free)
    usedF=set(i[1] for i in rF.values())
    sA=[(k,lb,ub-lb) for k,lb,ub,fo in sing if not fo]
    rA,uA=closer_match(sA,[i for i in allit if i[1] not in usedF])
    wF=sum(i[0]-lb for (k,lb,w) in sF for i in [rF.get(k)] if i)
    wA=sum(i[0]-lb for (k,lb,w) in sA for i in [rA.get(k)] if i)
    curF=sum(cur_w[k] for k,lb,w in sF); curA=sum(cur_w[k] for k,lb,w in sA)
    curM=sum(cur_w[b[0]] for b in bins if b[1]>10000)
    print(t,"singles F: match waste",wF,"cur",curF,"unmatched",len(uF),"| singles A: match waste",wA,"cur",curA,"unm",len(uA),"| multis cur waste",curM)
