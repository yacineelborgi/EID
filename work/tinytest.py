import json, sys, random, subprocess, time
sys.path.insert(0,'.'); sys.path.insert(0,'..')
from sol6 import State
import lns_rr
S=sys.argv[1]; typ=sys.argv[2]; nsmall=int(sys.argv[3]); nrand=int(sys.argv[4]); tlim=float(sys.argv[5]); seed=int(sys.argv[6])
inst=json.load(open("../datasets/6_shortage.json"))
st=State(inst,json.load(open(S+"/rr_shared.json"))); D=st.D; M=st.M
b=st.score()[0]
rng=random.Random(seed)
cov=st.covered()
kept=[(m,t) for m in st.used if m not in st.src for t in M[m]["dataRequirements"] if t==typ and (m,t) not in cov and st.bin_items.get((m,t))]
srcbins=[(s,typ) for s in st.src if next(iter(M[s]["dataRequirements"]))==typ]
allb=kept+srcbins
small=sorted((d["size"],d["id"]) for d in D if d["dataType"]==typ)[:nsmall]
smallset=set(i for s,i in small)
sel=set(k for k in allb if M[k[0]]["dataRequirements"][typ]["lowerBound"]<=1000)
sel|=set(k for k in allb if st.bin_items[k] & smallset)
rest=[k for k in allb if k not in sel]
sel|=set(rng.sample(rest, min(nrand,len(rest))))
refill=[];rh=[]
for (m,t) in sel:
    fo = st.is_full(m) if m not in st.src else False
    its=list(st.bin_items[(m,t)])
    for i in its: del st.item_bin[i]
    st.bin_items[(m,t)]=set(); refill.append([m,t,fo]); rh.append(its)
pin=f"{S}/tiny_in_{typ}.json"; ppar=f"{S}/tiny_p_{typ}.json"; pout=f"{S}/tiny_out_{typ}.json"
json.dump(st.to_sol(), open(pin,"w"))
json.dump({"multi_cands":lns_rr.LP_MULTI+lns_rr.LP_POS_MULTI,"extra_mono":lns_rr.LP_POS_MONO,"tlim":tlim,"refill":refill,"refill_hint":rh,"hint":{}}, open(ppar,"w"))
t0=time.time()
out=subprocess.run([sys.executable,"recreate.py","../datasets/6_shortage.json",pin,ppar,pout],capture_output=True,text=True)
st2=State(inst,json.load(open(pout)))
print("type",typ,"refill",len(refill),"before",b,"after",st2.score(),"time",round(time.time()-t0), out.stdout.strip().replace("\n"," | "), out.stderr[-300:], flush=True)
