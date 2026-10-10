import json, sys, numpy as np, collections
sys.path.insert(0, '.')
import mip6
inst = json.load(open("../datasets/6_shortage.json"))
M = inst["models"]; D = inst["datasets"]
mo = mip6.build(inst)
x = np.load("lp6_x.npy")
nM = mo.nM
f = x[mo.fo:mo.fo+nM]; hh = x[mo.ho:mo.ho+nM]; s = x[mo.so:mo.so+nM]
sol = json.load(open("../solutions/6_shortage_submission.json"))
srcs = {a for a,b in sol["modelMappings"]}
tg = collections.defaultdict(list)
for a,b in sol["modelMappings"]: tg[b].append(a)
used = set(m for _,m in sol["dataMappings"]) | srcs | set(tg)
cop = collections.defaultdict(int)
for db,m in sol["dataMappings"]: cop[m] |= D[db]["isCopyrighted"]
mode = {}
for m in used:
    mode[m] = "S" if m in srcs else ("H" if (cop[m] or tg[m]) else "F")
def desc(i):
    m=M[i]; r=m["dataRequirements"]
    return f"id{i} v{m['value']} E{m['energyCost']/1e9:.0f} " + " ".join(f"{t}[{q['lowerBound']},{q['upperBound']}]" for t,q in r.items())
print("--- LP vs current, differences (x>0.01 or used)")
rows=[]
for i in range(nM):
    lp = (f[i], hh[i], s[i])
    cur = mode.get(i, "-")
    curv = {"F":(1,0,0),"H":(0,1,0),"S":(0,0,1),"-":(0,0,0)}[cur]
    if max(abs(a-b) for a,b in zip(lp,curv))>0.01:
        rows.append((i, cur, lp))
for i,cur,lp in rows:
    print(cur, "LP F%.2f H%.2f S%.2f"%lp, desc(i))
print(len(rows))
