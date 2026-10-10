import json, sys, collections
inst=json.load(open('/home/user/EID/datasets/6_shortage.json'))
M=inst['models'];D=inst['datasets']
sol=json.load(open(sys.argv[1] if len(sys.argv)>1 else '/home/user/EID/solutions/6_shortage_submission.json'))
dm={d:m for d,m in sol['dataMappings']}
mm=sol['modelMappings']
us={s for s,_ in mm}
fill=collections.defaultdict(list)
for d,m in dm.items(): fill[(m,D[d]['dataType'])].append(D[d]['size'])
t='n'
rows=sorted((M[m]['dataRequirements'][t]['lowerBound'],M[m]['dataRequirements'][t]['upperBound'],sorted(v), m in us) for (m,tt),v in fill.items() if tt==t and M[m]['dataRequirements'][t]['lowerBound']<2500)
for r in rows: 
  if sum(r[2])-r[0]>50: print(r[0],r[1],r[2],sum(r[2])-r[0],'src' if r[3] else '')
