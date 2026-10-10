import json, collections
INSTP="/home/user/EID/datasets/2_medium.json"
TXT=open(INSTP).read(); I=json.loads(TXT); M=I['models']; D=I['datasets']; CAP=I['energyCap']
OUT="/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_lb/2_medium_submission.json"
def load(path=OUT):
    S=json.load(open(path))
    used=collections.defaultdict(list)
    for d,m in S['dataMappings']: used[m].append(d)
    src={s:t for s,t in S['modelMappings']}
    tg=collections.defaultdict(list)
    for s,t in S['modelMappings']: tg[t].append(s)
    trained=(set(used)|set(tg))-set(src)
    F=set();H=set()
    for m in trained:
        if m in tg or any(D[d]['isCopyrighted'] for d in used[m]): H.add(m)
        else: F.add(m)
    return S,used,src,tg,F,H
def energy(F,H,src): return sum(M[m]['energyCost'] for m in set(F)|set(H)|set(src))
def value(F,H): return sum(M[m]['value'] for m in F)+sum(M[m]['value']//2 for m in H)
