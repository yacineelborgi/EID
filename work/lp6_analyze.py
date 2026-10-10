import json, sys, time, numpy as np, collections
sys.path.insert(0, '.')
import mip6
inst = json.load(open("../datasets/6_shortage.json"))
M = inst["models"]
mo = mip6.build(inst)
h = mip6.to_highs(mo, integer=False)
h.run()
x = np.array(h.getSolution().col_value)
print("obj", h.getInfo().objective_function_value)
nM = mo.nM
f = x[mo.fo:mo.fo+nM]; hh = x[mo.ho:mo.ho+nM]; s = x[mo.so:mo.so+nM]
def frac(a): return ((a>1e-6)&(a<1-1e-6)).sum()
print("full sum", f.sum(), "frac", frac(f), "half", hh.sum(), frac(hh), "src", s.sum(), frac(s))
# compare with current solution
sol = json.load(open("../solutions/6_shortage_submission.json"))
D = inst["datasets"]
srcs = {a for a,b in sol["modelMappings"]}
tg = collections.defaultdict(list)
for a,b in sol["modelMappings"]: tg[b].append(a)
used = set(m for _,m in sol["dataMappings"]) | srcs | set(tg)
cop = collections.defaultdict(int)
for db,m in sol["dataMappings"]: cop[m] |= D[db]["isCopyrighted"]
cf=set(); ch=set()
for m in used:
    if m in srcs: continue
    (ch if (cop[m] or tg[m]) else cf).add(m)
print("cur full", len(cf), "half", len(ch), "src", len(srcs))
def ov(a, setx): 
    idx=np.array(sorted(setx)); return a[idx].sum()
print("LP mass on cur full", ov(f, cf), "on cur half", ov(hh, ch), "on cur src", ov(s, srcs))
# duals
dual = np.array(h.getSolution().row_dual)
for i,nm in enumerate(mo.names):
    if nm in ("energy",) or nm.startswith("cap_") or nm.startswith("src_"):
        print(nm, dual[i])
np.save("lp6_x.npy", x)
