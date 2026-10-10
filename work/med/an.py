import json,sys
import numpy as np
from collections import Counter,defaultdict
I=json.load(open('/home/user/EID/datasets/2_medium.json'))
S=json.load(open('/home/user/EID/solutions/2_medium_submission.json'))
M=I['models'];D=I['datasets']
T='ntic'
for t in T:
    fr=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
    cp=sorted(d['size'] for d in D if d['dataType']==t and d['isCopyrighted'])
    print(t,'free',len(fr),sum(fr),fr[:5],fr[-3:],'copy',len(cp),sum(cp),cp[:5],cp[-3:])
lbs=defaultdict(list); w=defaultdict(list)
for m in M:
    for t,r in m['dataRequirements'].items():
        lbs[t].append(r['lowerBound']); w[t].append(r['upperBound']-r['lowerBound'])
for t in T:
    a=np.array(lbs[t]); b=np.array(w[t])
    print(t,'lb pct',np.percentile(a,[0,10,50,90,100]),'width pct',np.percentile(b,[0,10,50,90,100]), 'neg width',(b<0).sum())
v=np.array([m['value'] for m in M]); e=np.array([m['energyCost'] for m in M])
print('value pct',np.percentile(v,[0,10,50,90,100]),'energy pct',np.percentile(e,[0,10,50,90,100]), 'cap',I['energyCap'], 'sum e',e.sum())
# solution analysis
dm=S['dataMappings']; mm=S['modelMappings']
srcs={s for s,_ in mm}
fill=defaultdict(lambda: defaultdict(int)); cpy=defaultdict(int)
for d,m in dm:
    fill[m][D[d]['dataType']]+=D[d]['size']; cpy[m]|=D[d]['isCopyrighted']
for s,tg in mm: cpy[tg]=1
used=set(fill)|{t for _,t in mm}
usedfree=defaultdict(int); usedtot=defaultdict(int); waste=defaultdict(int)
for d,m in dm:
    t=D[d]['dataType']; usedtot[t]+=D[d]['size']
    if not D[d]['isCopyrighted']: usedfree[t]+=D[d]['size']
for m in fill:
    for t,x in fill[m].items():
        r=M[m]['dataRequirements'][t]; waste[t]+=x-r['lowerBound']
for t in T: print(t,'usedfree',usedfree[t],'usedtot',usedtot[t],'waste',waste[t])
print('nfull',sum(1 for m in used if not cpy[m] and m not in srcs),'nhalf',sum(1 for m in used if cpy[m] and m not in srcs),'src',len(srcs))
