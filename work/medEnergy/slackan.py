import json,sys
p=json.load(open(sys.argv[1])); p=p[int(sys.argv[2]) if len(sys.argv)>2 else 0] if isinstance(p,list) else p
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']
full=set(p['full'])
for t in 'ntic':
  lbs=sorted(M[i]['dataRequirements'][t]['lowerBound'] for i in full if t in M[i]['dataRequirements'])
  items=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
  tot=sum(items); best=None
  for k in range(len(lbs)+1):
    cons=sum(lbs[k:])+max(sum(lbs[:k]), sum(items[:k]))
    # also each of first k bins consumes >= max(lb_j, item_j) sorted pairing
    cons2=sum(lbs[k:])+sum(max(a,b) for a,b in zip(lbs[:k],items[:k]))
    s=tot-max(cons,cons2)
    if best is None or s<best[0]: best=(s,k)
  print(t,'nbins',len(lbs),'slack',tot-sum(lbs),'eff slack',best, 'nsmall<=10000',sum(1 for l in lbs if l<=10000))
