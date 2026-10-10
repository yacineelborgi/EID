import json,sys
sys.path.insert(0,'/home/user/EID')
from ptype import type_data, M, D, INST
import numpy as np
from collections import Counter
plan=json.load(open(sys.argv[1]))
sol=json.load(open(sys.argv[2]))
dm={d:m for d,m in sol['dataMappings']}
for t in 'ntic':
    bins,items=type_data(plan,t)
    L=np.array([b[1] for b in bins]); U=np.array([b[2] for b in bins])
    print(t,'L quantiles',np.percentile(L,[0,10,25,50,75,90,100]).astype(int),'width med',int(np.median(U-L)), 'rel width med',np.median((U-L)/np.maximum(L,1)))
    cnt=Counter(); per={}
    for d,m in dm.items():
        if D[d]['dataType']==t: per.setdefault(m,[]).append(D[d]['size'])
    for b in bins:
        cnt[len(per.get(b[0],[]))]+=1
    print('  items per bin',sorted(cnt.items()))
    # waste by L bucket
    wb=Counter()
    for b in bins:
        if b[0] in per:
            s=sum(per[b[0]]); wb[min(b[1]//3000,8)]+=s-b[1]
    print('  waste by L/3000 bucket',sorted(wb.items()))
