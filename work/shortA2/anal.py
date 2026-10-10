import json,sys
inst=json.load(open('/home/user/EID/datasets/6_shortage.json'))
M,D=inst['models'],inst['datasets']
p=json.load(open(sys.argv[1]))
full=set(p['full']);half=set(p['half']);src=set(p['source']);sr={tuple(x) for x in p['srcreq']}
print('full',len(full),'half',len(half),'src',len(src),'srcreq',len(sr))
val=sum(M[i]['value'] for i in full)+sum(M[i]['value']//2 for i in half)
E=sum(M[i]['energyCost'] for i in full|half|src)
print('value',val,'energy slack',inst['energyCap']-E)
for t in 'ntic':
    fr=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
    cp=sorted(d['size'] for d in D if d['dataType']==t and d['isCopyrighted'])
    al=sorted(fr+cp)
    bF=sorted((M[i]['dataRequirements'][t]['lowerBound'],M[i]['dataRequirements'][t]['upperBound']) for i in full if t in M[i]['dataRequirements'])
    bH=sorted((M[i]['dataRequirements'][t]['lowerBound'],M[i]['dataRequirements'][t]['upperBound']) for i in half|src if t in M[i]['dataRequirements'] and (i,t) not in sr)
    # multi-threshold slack: for each threshold th, sum lb of bins>th + sum smallest k items (k=#bins<=th) <= cap
    def slack(bins,items):
        best=None
        out=[]
        for th in [0,1,10,30,100,200,300,500,800,1000,1500,2000,3000,5000,10000]:
            k=sum(1 for l,u in bins if l<=th)
            big=sum(l for l,u in bins if l>th)
            s=sum(items[:k])+big
            out.append((th,k,sum(items)-s))
        return out
    print(t,'F bins',len(bF),'sumlb',sum(l for l,u in bF),'freecap',sum(fr),'nfree',len(fr))
    print('  F slack by th',[(th,k,s) for th,k,s in slack(bF,fr)])
    allb=bF+bH
    print('  A bins',len(allb),'sumlb',sum(l for l,u in allb),'cap',sum(al),'n',len(al))
    print('  A slack',[(th,k,s) for th,k,s in slack(allb,al)])
