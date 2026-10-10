import json, sys, collections
sys.path.insert(0,'.')
from sol6 import State
from sa_drive import run_sa
inst=json.load(open("../datasets/6_shortage.json")); sol=json.load(open("../solutions/6_shortage_submission.json"))
st=State(inst,sol); D=st.D; M=st.M
t=sys.argv[1]; K0=float(sys.argv[2]); K1=float(sys.argv[3]); sinks=[int(x) for x in sys.argv[4:]]
deficit,pool,newbin,bins=run_sa(st,t,sinks,int(1e8),3,10,0.1,K0=K0,K1=K1)
fill=collections.Counter(); cnt=collections.Counter()
for i,b in newbin.items():
    if b is not None: fill[b]+=D[i]["size"]; cnt[b]+=1
agg=collections.defaultdict(lambda:[0,0,0])
for k,lb,ub,fo in bins:
    f=fill[k]
    cls=("F" if fo else "A")+("tiny" if lb<=1000 else "single" if lb<=10000 else "multi")
    if f<lb: agg[cls][2]+=1
    else: agg[cls][0]+=1; agg[cls][1]+=f-lb
print("deficit",deficit,"pool",pool)
for k,v in sorted(agg.items()): print(k,"n",v[0],"waste",v[1],"unmet",v[2])
