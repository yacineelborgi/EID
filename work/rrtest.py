import json, sys, random, subprocess, time
sys.path.insert(0,'.'); sys.path.insert(0,'..')
from sol6 import State
import lns_rr
S=sys.argv[1]; seed=int(sys.argv[2]); typ=sys.argv[3]
inst=json.load(open("../datasets/6_shortage.json"))
st=State(inst,json.load(open(S+"/rr_shared.json")))
b=st.score()[0]
rng=random.Random(seed)
rng2=random.Random(0)
class R:  # force energy-targeted mixed ruin
    def __getattr__(s,a): return getattr(rng,a)
    def random(s): return 0.0
touched,dropped=lns_rr.ruin(st, R(), [typ], 3,2,1, mode="mixed")
M=st.M
print("dropped", len(dropped), "multi tg", sum(1 for m in dropped if len(M[m]["dataRequirements"])>1), "lost", b-st.score()[0])
pin=f"{S}/rrt_in.json"; ppar=f"{S}/rrt_p.json"; pout=f"{S}/rrt_out.json"
json.dump(st.to_sol(), open(pin,"w"))
json.dump({"multi_cands":[m for m in dropped if len(M[m]["dataRequirements"])>1]+lns_rr.LP_MULTI+lns_rr.LP_POS_MULTI,"extra_mono":[m for m in dropped if len(M[m]["dataRequirements"])==1]+lns_rr.LP_POS_MONO,"tlim":120,"hint":{str(k):[v[0],v[1]] for k,v in st._hint.items()}}, open(ppar,"w"))
out=subprocess.run([sys.executable,"recreate.py","../datasets/6_shortage.json",pin,ppar,pout],capture_output=True,text=True)
st2=State(inst,json.load(open(pout)))
print("before",b,"after",st2.score(), out.stdout.strip().replace("\n"," | "), out.stderr[-300:])
