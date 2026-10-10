import json,sys
I=json.load(open('/home/user/EID/datasets/2_medium.json')); M=I['models']; D=I['datasets']
plans=json.load(open(sys.argv[1])); 
if isinstance(plans,dict): plans=[plans]
for k in ([int(sys.argv[2])] if len(sys.argv)>2 else range(len(plans))):
    p=plans[k]; full=p['full']
    E=sum(M[i]['energyCost'] for i in p['full']+p['half']+p['source'])
    out=[]
    for t in 'ntic':
        fr=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
        lbs=sorted(M[i]['dataRequirements'][t]['lowerBound'] for i in full if t in M[i]['dataRequirements'])
        # greedy lower bound on consumption: assign smallest items to smallest bins (one item each) for bins with lb < item
        # multi-threshold: max over th of sum(lb>th)+S(#lb<=th)
        best=0
        import bisect
        P=[0]
        for s in fr: P.append(P[-1]+s)
        for j in range(len(lbs)+1):
            th_cnt=j  # bins with smallest j lbs
            v=sum(lbs[j:])+ (P[j] if j<len(P) else 10**18)
            # valid only as max over thresholds: smallest j bins consume >= S(j) and >= their lbs
            v=sum(lbs[j:])+max(P[j] if j<len(P) else 10**18, sum(lbs[:j]))
            best=max(best,v)
        out.append((t,len(lbs),sum(fr)-best))
    print(k,p.get('obj'),'Eslack',I['energyCap']-E,'free slack (multi-th)',out)
