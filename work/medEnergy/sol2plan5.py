import json, sys
from collections import defaultdict
inst=json.load(open('/home/user/EID/datasets/5_energy.json')); M=inst['models']; D=inst['datasets']
S=json.load(open(sys.argv[1]))
dm={d:m for d,m in S['dataMappings']}
srcs={a for a,b in S['modelMappings']}
srq=[[b, next(iter(M[a]['dataRequirements']))] for a,b in S['modelMappings']]
tg={b for a,b in S['modelMappings']}
cp=defaultdict(int)
for d,m in dm.items(): cp[m]|=D[d]['isCopyrighted']
trained=(set(dm.values())-srcs)|tg
half=sorted(i for i in trained if cp[i] or i in tg); full=sorted(set(trained)-set(half))
p={'full':full,'half':half,'source':sorted(srcs),'srcreq':srq}
p['obj']=sum(M[i]['value'] for i in full)+sum(M[i]['value']//2 for i in half)
json.dump(p,open(sys.argv[2],'w')); print('plan value',p['obj'])
