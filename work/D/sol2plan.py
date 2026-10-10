import json
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']
def sol2plan(S):
    dm=S['dataMappings']; mm=S['modelMappings']
    srcs={a for a,b in mm}; tg={}
    for a,b in mm: tg.setdefault(b,[]).append(a)
    by={}
    for d,m in dm: by.setdefault(m,[]).append(d)
    trained=(set(by)|set(tg))-srcs
    full=[];half=[];srcreq=[]
    for m in sorted(trained):
        cop=any(D[d]['isCopyrighted'] for d in by.get(m,[]))
        if cop or m in tg: half.append(m)
        else: full.append(m)
        for a in tg.get(m,[]): srcreq.append([m,list(M[a]['dataRequirements'])[0]])
    fills={}
    for m in full:
        for d in by.get(m,[]): fills.setdefault(D[d]['dataType'],{}).setdefault(m,[]).append(d)
    return {'full':full,'half':half,'source':sorted(srcs),'srcreq':srcreq,'obj':sum(M[i]['value'] for i in full)+sum(M[i]['value']//2 for i in half)}, fills, by
if __name__=='__main__':
    import sys
    S=json.load(open(sys.argv[1])); p,f,by=sol2plan(S); print('obj',p['obj'],len(p['full']),len(p['half']),len(p['source']))
    E=sum(M[i]['energyCost'] for i in p['full']+p['half']+p['source']); print('energy slack',inst['energyCap']-E)
    for t in 'ntic':
        fr=[d for d in D if d['dataType']==t and not d['isCopyrighted']]; cap=sum(d['size'] for d in fr)
        usedF=0; sumL=0; wl=0; ws=0; ns=0
        for m in p['full']:
            if t not in M[m]['dataRequirements']: continue
            L=M[m]['dataRequirements'][t]['lowerBound']; c=sum(D[d]['size'] for d in f.get(t,{}).get(m,[]))
            usedF+=c; sumL+=L
            if L<=3000: ws+=c-L; ns+=1
            else: wl+=c-L
        usedFh=sum(D[d]['size'] for m in p['half'] for d in by.get(m,[]) if D[d]['dataType']==t and not D[d]['isCopyrighted'])
        print(t,'cap',cap,'sumL',sumL,'small n',ns,'waste small',ws,'large',wl,'free used by half',usedFh,'unused',cap-usedF-usedFh)
    json.dump(p,open('plan320.json','w'))
