import json,sys,numpy as np
inst=json.load(open('/home/user/EID/datasets/6_shortage.json'))
M,D=inst['models'],inst['datasets']
p=json.load(open(sys.argv[1]))
full=set(p['full']);half=set(p['half']);src=set(p['source']);sr={tuple(x) for x in p['srcreq']}
for t in sys.argv[2]:
    fr=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
    cp=sorted(d['size'] for d in D if d['dataType']==t and d['isCopyrighted'])
    bF=sorted((M[i]['dataRequirements'][t]['lowerBound'],M[i]['dataRequirements'][t]['upperBound']) for i in full if t in M[i]['dataRequirements'])
    bH=sorted((M[i]['dataRequirements'][t]['lowerBound'],M[i]['dataRequirements'][t]['upperBound']) for i in half|src if t in M[i]['dataRequirements'] and (i,t) not in sr)
    print(t,'free items',len(fr),'small',fr[:60])
    print(t,'copy items',len(cp),'small',cp[:60])
    print('F bins lb hist',np.histogram([l for l,u in bF],bins=[0,1,100,300,500,1000,2000,3000,5000,8000,10000,15000,20000,30000,50000,110000])[0])
    print('F tiny',[b for b in bF if b[0]<=3000])
    print('H bins lb hist',np.histogram([l for l,u in bH],bins=[0,1,100,300,500,1000,2000,3000,5000,8000,10000,15000,20000,30000,50000,110000])[0])
    print('H tiny',[b for b in bH if b[0]<=3000])
    print('item hist free',np.histogram(fr,bins=[0,100,300,500,1000,2000,3000,5000,8000,10001])[0])
    print('item hist copy',np.histogram(cp,bins=[0,100,300,500,1000,2000,3000,5000,8000,10001])[0])
