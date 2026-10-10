"""Volume compaction of one type with the HiGHS MIP (resid_mip vol mode): release tiny bins,
holders of the smallest datasets and some random bins, refill them exactly while minimizing
the volume used (freed datasets go back to the pool). Models are unchanged.
usage: python volcompact.py inst sol_in type nsmall nrand tlim seed sol_out"""
import json, sys, random, subprocess, os
sys.path.insert(0, '.')
from sol6 import State

inst_p, sol_p, typ = sys.argv[1], sys.argv[2], sys.argv[3]
nsmall, nrand, tlim, seed, out = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
inst = json.load(open(inst_p))
st = State(inst, json.load(open(sol_p)))
D, M = st.D, st.M
rng = random.Random(seed)
cov = st.covered()
bins = [(m, typ) for m in st.used if typ in M[m]["dataRequirements"] and (m in st.src or (m, typ) not in cov)
        and st.bin_items.get((m, typ))]
small = set(i for _, i in sorted((d["size"], d["id"]) for d in D if d["dataType"] == typ)[:nsmall])
sel = set(k for k in bins if M[k[0]]["dataRequirements"][typ]["lowerBound"] <= 1000 or st.bin_items[k] & small)
rest = [k for k in bins if k not in sel]
sel |= set(rng.sample(rest, min(nrand, len(rest))))
pool0 = sum(d["size"] for d in D if d["dataType"] == typ and d["id"] not in st.item_bin)
refill, rh = [], []
fo_of = {k: (st.is_full(k[0]) if k[0] not in st.src else False) for k in sel}
for k in sel:
    its = list(st.bin_items[k])
    for i in its:
        del st.item_bin[i]
    st.bin_items[k] = set()
    refill.append([k[0], typ, fo_of[k]]); rh.append(its)
base = out[:-5]
json.dump(st.to_sol(), open(base + "_in.json", "w"))
json.dump({"refill": refill, "refill_hint": rh, "vol": True, "tlim": tlim, "hint": {}}, open(base + "_p.json", "w"))
r = subprocess.run([sys.executable, "resid_mip.py", inst_p, base + "_in.json", base + "_p.json", out], capture_output=True, text=True)
st2 = State(inst, json.load(open(out)))
pool1 = sum(d["size"] for d in D if d["dataType"] == typ and d["id"] not in st2.item_bin)
freeP = sum(d["size"] for d in D if d["dataType"] == typ and d["id"] not in st2.item_bin and not d["isCopyrighted"])
print(typ, "released", len(refill), "pool", pool0, "->", pool1, "(free", freeP, ")", st2.score(), r.stdout.strip(), flush=True)
