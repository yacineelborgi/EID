"""Decompose a solution's value as L(u) - sum(dev) - u.slack under relaxation R."""
import sys, json, pickle
sys.path.insert(0,'.')
from rel import *
from collections import defaultdict
inst = load('/home/user/EID/datasets/2_medium.json')
M=inst['models']; D=inst['datasets']
S = json.load(open(sys.argv[1]))
d = pickle.load(open('lp1.pkl','rb')); u=d['u']; L=d['L']; red=d['red']
R = Rel(inst, tau={t:1 for t in TYPES}); R.build()
srcs={s for s,_ in S['modelMappings']}
tg=defaultdict(set)
for s,t in S['modelMappings']: tg[t].add(next(iter(M[s]['dataRequirements'])))
cp=defaultdict(int); used=set()
fill=defaultdict(lambda: defaultdict(int))
for dd,m in S['dataMappings']:
    used.add(m); cp[m]|=D[dd]['isCopyrighted']; fill[m][D[dd]['dataType']]+=D[dd]['size']
used|=set(tg)
opt={}
for i in range(R.nM):
    if i in srcs: opt[i]='source'
    elif i in used:
        if i in tg or cp[i]:
            Z=''.join(t for t in M[i]['dataRequirements'] if t in tg[i]); opt[i]='half:'+Z
        else: opt[i]='full'
    else: opt[i]='off'
dev=0; devlist=[]
for i,ops in enumerate(red):
    best=max(ops)[0]
    p=[pp for pp,n in ops if n==opt[i]][0]
    if best-p>0: devlist.append((float(best-p),i,opt[i],max(ops)[1]))
    dev+=best-p
val=sum(M[i]['value'] for i in used if opt[i]=='full')+sum(M[i]['value']//2 for i in used if opt[i].startswith('half'))
print('value',val,'L',float(L),'sum dev',float(dev),'L-dev',float(L-dev),'=> slack cost',float(L-dev)-val)
devlist.sort(reverse=True)
print('ndev',len(devlist)); print(devlist[:40])
# slack analysis
E=sum(M[i]['energyCost'] for i in used|srcs)
print('energy slack',inst['energyCap']-E,'cost',float(u['E'])*(inst['energyCap']-E))
for t in TYPES:
    fr=sum(D[dd]['size'] for dd,m in S['dataMappings'] if D[dd]['dataType']==t and not D[dd]['isCopyrighted'] and opt[m]=='full')
    frall=sum(D[dd]['size'] for dd,m in S['dataMappings'] if D[dd]['dataType']==t and not D[dd]['isCopyrighted'])
    k=sum(1 for i in used if opt[i]=='full' and t in M[i]['dataRequirements'] and M[i]['dataRequirements'][t]['lowerBound']<=1)
    nontiny_lb=sum(M[i]['dataRequirements'][t]['lowerBound'] for i in used if opt[i]=='full' and t in M[i]['dataRequirements'] and M[i]['dataRequirements'][t]['lowerBound']>1)
    nontiny_fill=sum(fill[i][t] for i in used if opt[i]=='full' and t in M[i]['dataRequirements'] and M[i]['dataRequirements'][t]['lowerBound']>1)
    tiny_fill=sum(fill[i][t] for i in used if opt[i]=='full' and t in M[i]['dataRequirements'] and M[i]['dataRequirements'][t]['lowerBound']<=1)
    print(t,'F',R.F[t],'free used by full',fr,'free used by others',frall-fr,'unused',R.F[t]-frall,'k tiny',k,'P(k)',int(R.P[t][k]),'tiny fill',tiny_fill,'nontiny waste',nontiny_fill-nontiny_lb, 'sources',sum(1 for i in srcs if t in M[i]['dataRequirements']),'srcreq',sum(1 for i in tg if t in tg[i]))
