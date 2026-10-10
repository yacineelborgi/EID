import json, sys
sys.path.insert(0,'.')
from sol6 import State
from sa_drive import run_sa
inst=json.load(open("../datasets/6_shortage.json")); sol=json.load(open("../solutions/6_shortage_submission.json"))
st=State(inst,sol); D=st.D; M=st.M
t=sys.argv[1]; sinks=[int(x) for x in sys.argv[2:]]
deficit,pool,newbin,bins=run_sa(st,t,sinks,int(1e8),3,10,0.1)
fill={}
for i,b in newbin.items():
    if b is not None: fill[b]=fill.get(b,0)+D[i]["size"]
print("deficit",deficit,"pool",pool)
for k,lb,ub,fo in bins:
    f=fill.get(k,0)
    if f<lb:
        its=sorted(D[i]["size"] for i,b in newbin.items() if b==k)
        print("DEF bin",k,"lb",lb,"ub",ub,"fo",fo,"fill",f,"items",its,"src" if k[0] in st.src else "", "E", M[k[0]]["energyCost"])
pl=sorted((D[i]["size"], D[i]["isCopyrighted"]) for i,b in newbin.items() if b is None)
print("pool items", pl)
