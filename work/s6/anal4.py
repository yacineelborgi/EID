import json, sys, collections
sys.path.insert(0,'.')
from enc import classify
inst=json.load(open('/home/user/EID/datasets/6_shortage.json'))
M=inst['models'];D=inst['datasets']
sol=json.load(open(sys.argv[1] if len(sys.argv)>1 else '/home/user/EID/solutions/6_shortage_submission.json'))
dm, mm, full, half, us = classify(inst, sol)
fill=collections.defaultdict(list)
for d,m in dm.items(): fill[(m,D[d]['dataType'])].append(D[d]['size'])
for t in 'ntic':
  al=sorted(d['size'] for d in D if d['dataType']==t)
  rows=sorted((M[m]['dataRequirements'][t]['lowerBound'],m) for (m,tt),v in fill.items() if tt==t)
  # assortative excess per bin
  ex=[]
  for j,(L,m) in enumerate(rows):
    use=sum(fill[(m,t)]); g=max(L,al[j])
    ex.append((use-g, L, m, 'F' if m in full else ('S' if m in us else 'H'), len(fill[(m,t)]), M[m]['dataRequirements'][t]['upperBound']-L))
  tot=sum(e[0] for e in ex)
  big=[e for e in ex if e[0]>300]
  print(t,'total excess over assortative',tot,'n>300',len(big),'sum',sum(e[0] for e in big))
  print('  ', sorted(big,reverse=True)[:12])
