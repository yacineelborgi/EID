import json,sys
inst=json.load(open('/home/user/EID/datasets/5_energy.json'))
M,D=inst['models'],inst['datasets']
sol=json.load(open(sys.argv[1]))
dm={d:m for d,m in sol['dataMappings']}
src={s:t for s,t in sol['modelMappings']}
tg={}
for s,t in sol['modelMappings']: tg.setdefault(t,[]).append(next(iter(M[s]['dataRequirements'])))
used=set(dm.values())|set(src)|set(tg)
defiled={i:(i in tg) for i in used}
for d,m in dm.items():
    if D[d]['isCopyrighted']: defiled[m]=True
for t in 'ntic':
    fr=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
    lbs=sorted(M[i]['dataRequirements'][t]['lowerBound'] for i in used if i not in src and not defiled[i] and t in M[i]['dataRequirements'])
    w=sum(max(l,s)-l for l,s in zip(lbs,fr))
    # actual waste of tiny bins
    print(t,'nbinsA',len(lbs),'min waste A',w)
