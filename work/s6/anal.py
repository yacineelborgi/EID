import json, sys, collections
sys.path.insert(0,'/home/user/EID')
inst=json.load(open('/home/user/EID/datasets/6_shortage.json'))
M=inst['models'];D=inst['datasets']
sol=json.load(open(sys.argv[1] if len(sys.argv)>1 else '/home/user/EID/solutions/6_shortage_submission.json'))
dm={d:m for d,m in sol['dataMappings']}
mm=sol['modelMappings']
us={s for s,_ in mm}; tg={t for _,t in mm}
trained=(set(dm.values())|tg)-us
cop={m for d,m in dm.items() if D[d]['isCopyrighted']}
full={i for i in trained if i not in cop and i not in tg}
half=trained-full
fill=collections.defaultdict(list)
for d,m in dm.items(): fill[(m,D[d]['dataType'])].append(D[d]['size'])
for t in 'ntic':
  fs=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
  al=sorted(d['size'] for d in D if d['dataType']==t)
  F=sum(fs); T=sum(al)
  freeuse=sum(D[d]['size'] for d,m in dm.items() if D[d]['dataType']==t and not D[d]['isCopyrighted'])
  rows=[(m,M[m]['dataRequirements'][t]['lowerBound'],sum(v),len(v), m in full) for (m,tt),v in fill.items() if tt==t]
  fullrows=[r for r in rows if r[4]]
  lbF=sum(r[1] for r in fullrows); useF=sum(r[2] for r in fullrows)
  lbA=sum(r[1] for r in rows); useA=sum(r[2] for r in rows)
  small=[r for r in rows if r[1]<2500]
  print(t,'F',F,'full bins',len(fullrows),'lb',lbF,'use',useF,'waste',useF-lbF,'| free used by nonfull',freeuse-useF, '| T',T,'all bins',len(rows),'lb',lbA,'use',useA,'waste',useA-lbA,'unused',T-useA)
  print('   small bins',len(small),'lb',sum(r[1] for r in small),'use',sum(r[2] for r in small),'S(k) all',sum(al[:len(small)]),
        ' big waste', sum(r[2]-r[1] for r in rows if r[1]>=2500), 'nbig', len(rows)-len(small))
  w=sorted([r[2]-r[1] for r in rows if r[1]>=2500],reverse=True)
  print('   top big wastes', w[:15])
