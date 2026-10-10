import json, sys
from locs import *
p=copyp(json.load(open(sys.argv[1]))); t=sys.argv[2]; maxloss=int(sys.argv[3])
base=J(p)
print('base',base)
res=[]
for i in p['full']:
    rq=M[i]['dataRequirements']
    if t not in rq: continue
    for kind in ('dm','rm'):
        loss=M[i]['value']-M[i]['value']//2 if kind=='dm' else M[i]['value']
        if loss>maxloss: continue
        q=demote(p,i) if kind=='dm' else remove(p,i)
        Js=J(q,t)
        res.append((min(Js[t][1],Js[t][2]),kind,i,loss,Js[t],{tt:rq[tt]['lowerBound'] for tt in rq}))
res.sort(reverse=True)
for x in res[:25]: print(x)
