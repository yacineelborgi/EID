import json, sys
sys.path.insert(0, '/home/user/EID')
from solve import score_of, strict_ok
TYPES='ntic'
txt=open('/home/user/EID/datasets/7_big.json').read()
inst=json.loads(txt); M,D=inst['models'],inst['datasets']; cap=inst['energyCap']
sol=json.load(open(sys.argv[1]))
sc,ok,msg=score_of(txt,sol)
print('score',sc,ok,msg,'strict',strict_ok(inst,sol))
dm={}
for d,m in sol['dataMappings']: dm.setdefault(m,[]).append(d)
srcs={s for s,_ in sol['modelMappings']}; tg={}
for s,t in sol['modelMappings']: tg.setdefault(t,[]).append(s)
used=set(dm)|srcs|set(tg)
E=sum(M[i]['energyCost'] for i in used)
print('energy',E,'cap',cap,'slack',cap-E)
full=[i for i in used if i not in srcs and i not in tg and all(not D[d]['isCopyrighted'] for d in dm.get(i,[]))]
half=[i for i in used if i not in srcs and i not in full]
print('full',len(full),'half',len(half),'src',len(srcs))
# data waste
usedd={d for d,m in sol['dataMappings']}
for t in TYPES:
  for c in (0,1):
    tot=sum(d['size'] for d in D if d['dataType']==t and d['isCopyrighted']==c)
    u=sum(D[d]['size'] for d in usedd if D[d]['dataType']==t and D[d]['isCopyrighted']==c)
    print(t,c,'total',tot,'used',u,'left',tot-u, 'nleft', sum(1 for d in D if d['dataType']==t and d['isCopyrighted']==c and d['id'] not in usedd))
# waste per bin
w={t:0 for t in TYPES}; lbsum={t:0 for t in TYPES}
for i,ds in dm.items():
  for t in TYPES:
    s=sum(D[d]['size'] for d in ds if D[d]['dataType']==t)
    if s: w[t]+=s-M[i]['dataRequirements'][t]['lowerBound']; lbsum[t]+=M[i]['dataRequirements'][t]['lowerBound']
print('waste',w,'lbsum',lbsum)
