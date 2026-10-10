import json,sys
I=json.load(open('/home/user/EID/datasets/2_medium.json')); M=I['models']; D=I['datasets']
def stats(p):
  sr=set(map(tuple,p['srcreq'])); out=[]
  for t in 'ntic':
    F=[(M[m]['dataRequirements'][t]['lowerBound'],M[m]['dataRequirements'][t]['upperBound']) for m in p['full'] if t in M[m]['dataRequirements']]
    H=[(M[m]['dataRequirements'][t]['lowerBound']) for m in p['half']+p['source'] if t in M[m]['dataRequirements'] and (m,t) not in sr]
    fs=sum(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
    cs=sum(d['size'] for d in D if d['dataType']==t and d['isCopyrighted'])
    out.append((t,len(F),fs-sum(a for a,b in F),sum(1 for a,b in F if a<=3000),len(H),fs+cs-sum(a for a,b in F)-sum(H)))
  E=I['energyCap']-sum(M[m]['energyCost'] for m in p['full']+p['half']+p['source'])
  val=sum(M[m]['value'] for m in p['full'])+sum(M[m]['value']//2 for m in p['half'])
  return val,E,out
if __name__=='__main__':
  ps=json.load(open(sys.argv[1])); ps=[ps] if isinstance(ps,dict) else ps
  for k,p in enumerate(ps):
    v,E,o=stats(p); print(k,p.get('obj'),v,'E',E,' '.join(f'{t}:F{n} fs{s} sm{sm} H{h} tot{ts}' for t,n,s,sm,h,ts in o))
