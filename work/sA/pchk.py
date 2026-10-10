"""Per-type F slack of a plan with exact single-item small-bin matching. usage: pchk.py plan.json|sol.json [idx] [theta]"""
import sys, json, numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); M=inst['models']; D=inst['datasets']
def full_of(o):
    if 'full' in o: return set(o['full'])
    cp=defaultdict(int); used=set()
    for d,m in o['dataMappings']: used.add(m); cp[m]|=D[d]['isCopyrighted']
    tg={t for _,t in o['modelMappings']}; srcs={s for s,_ in o['modelMappings']}
    return {i for i in used if not cp[i] and i not in tg and i not in srcs}
def check(full, theta=3000):
    res={}
    for t in 'ntic':
        items=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
        cap=sum(items); sb=[]; big=0
        for i in full:
            r=M[i]['dataRequirements'].get(t)
            if not r: continue
            if r['lowerBound']<=theta: sb.append((r['lowerBound'],r['upperBound']))
            else: big+=r['lowerBound']
        its=[s for s in items if s<=max([u for _,u in sb]+[0])]
        C=np.full((len(sb),len(its)),1e9)
        for a,(L,U) in enumerate(sb):
            for q,s in enumerate(its):
                if L<=s<=U: C[a,q]=s
        if sb:
            ri,ci=linear_sum_assignment(C); mc=C[ri,ci].sum()
        else: mc=0
        res[t]=(len(sb), int(cap-big-mc) if mc<1e9 else 'INF', int(cap-big-sum(L for L,_ in sb)))
    return res
if __name__=='__main__':
    o=json.load(open(sys.argv[1]))
    if isinstance(o,list): o=o[int(sys.argv[2]) if len(sys.argv)>2 else -1]
    th=int(sys.argv[3]) if len(sys.argv)>3 else 3000
    print(o.get('obj'), check(full_of(o),th))
