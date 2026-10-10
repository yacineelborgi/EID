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
    F=sum(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
    T=sum(d['size'] for d in D if d['dataType']==t)
    useA=0;lbA=0;useB=0;lbB=0;freeB=0
    fill={}
    for d,m in dm.items():
        if D[d]['dataType']==t: fill[m]=fill.get(m,0)+D[d]['size']
    for m,s in fill.items():
        lb=M[m]['dataRequirements'][t]['lowerBound']
        if not defiled.get(m) and m not in src: useA+=s;lbA+=lb
        else: useB+=s;lbB+=lb
    freeused=sum(D[d]['size'] for d in dm if D[d]['dataType']==t and not D[d]['isCopyrighted'])
    allused=sum(D[d]['size'] for d in dm if D[d]['dataType']==t)
    print(t,'A use',useA,'lb',lbA,'waste',useA-lbA,'freeLeft',F-useA,'| B use',useB,'lb',lbB,'waste',useB-lbB,'| unused total',T-allused, 'unused free', F-freeused)
val=sum(M[i]['value']//(2 if defiled[i] else 1) for i in used if i not in src)
print('value',val,'n full',sum(1 for i in used if i not in src and not defiled[i]),'half',sum(1 for i in used if i not in src and defiled[i]),'src',len(src))
