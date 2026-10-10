import json, sys, random, subprocess, time
sys.path.insert(0,'.'); sys.path.insert(0,'..')
from sol6 import State
import lns_rr
S=sys.argv[1]; nref=int(sys.argv[2]); typ=sys.argv[3]; seed=int(sys.argv[4]); tlim=float(sys.argv[5])
inst=json.load(open("../datasets/6_shortage.json"))
st=State(inst,json.load(open(S+"/rr_shared.json")))
b=st.score()[0]
rng=random.Random(seed)
touched,dropped=lns_rr.ruin(st, rng, [typ], 6,3,2, mode="mixed")
cov=st.covered()
kept=[(m,t) for m in st.used if m not in st.src for t in st.M[m]["dataRequirements"] if t==typ and (m,t) not in cov and st.bin_items.get((m,t))]
refill=[];rh=[]
for (m,t) in rng.sample(kept, min(nref,len(kept))):
    fo=st.is_full(m); its=list(st.bin_items[(m,t)])
    for i in its: del st.item_bin[i]
    st.bin_items[(m,t)]=set(); refill.append([m,t,fo]); rh.append(its)
pin=f"{S}/big_in_{seed}.json"; ppar=f"{S}/big_p_{seed}.json"; pout=f"{S}/big_out_{seed}.json"
json.dump(st.to_sol(), open(pin,"w"))
M=st.M
json.dump({"multi_cands":[m for m in dropped if len(M[m]["dataRequirements"])>1]+lns_rr.LP_MULTI,"extra_mono":[m for m in dropped if len(M[m]["dataRequirements"])==1],"tlim":tlim,"refill":refill,"refill_hint":rh,"hint":{str(k):[v[0],v[1]] for k,v in st._hint.items()}}, open(ppar,"w"))
t0=time.time()
out=subprocess.run([sys.executable,"recreate.py","../datasets/6_shortage.json",pin,ppar,pout],capture_output=True,text=True)
st2=State(inst,json.load(open(pout)))
print("nref",len(refill),"type",typ,"before",b,"after",st2.score(),"time",round(time.time()-t0), out.stdout.strip().replace("\n"," | "), flush=True)
