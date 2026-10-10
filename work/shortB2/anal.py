import json, sys, collections
inst = json.load(open('/home/user/EID/datasets/6_shortage.json')); M, D = inst['models'], inst['datasets']
sol = json.load(open(sys.argv[1]))
dm = sol['dataMappings']; mm = sol['modelMappings']
src = {a for a, b in mm}; tg = {b for a, b in mm}
per = collections.defaultdict(list)
for d, m in dm: per[(m, D[d]['dataType'])].append(d)
trained = set(m for (m, t) in per) - src | tg
val = 0
for i in trained:
    fr = all(not D[d]['isCopyrighted'] for (m, t), ds in per.items() if m == i for d in ds) and i not in tg
    val += M[i]['value'] if fr else M[i]['value'] // 2
E = sum(M[i]['energyCost'] for i in trained | src)
print('val', val, 'ntrained', len(trained), 'nsrc', len(src), 'Eslack', inst['energyCap'] - E)
used = {d for d, m in dm}
for t in 'ntic':
    al = sorted(x['size'] for x in D if x['dataType'] == t)
    bins = [(m, M[m]['dataRequirements'][t]['lowerBound'], sum(D[d]['size'] for d in ds), len(ds)) for (m, tt), ds in per.items() if tt == t]
    Ls = sorted(b[1] for b in bins)
    G = sum(max(a, b) for a, b in zip(Ls, al))
    sl = sum(b[1] for b in bins); sc = sum(b[2] for b in bins)
    left = sum(x['size'] for x in D if x['dataType'] == t and x['id'] not in used)
    nl = sum(1 for x in D if x['dataType'] == t and x['id'] not in used)
    w1 = sum(b[2] - b[1] for b in bins if b[3] == 1); wm = sum(b[2] - b[1] for b in bins if b[3] > 1)
    nb1 = sum(1 for b in bins if b[3] == 1)
    print(t, 'nb', len(bins), 'n1', nb1, 'sumL', sl, 'cons', sc, 'G', G, 'G-sumL', G - sl, 'waste', sc - sl, 'w1', w1, 'wmulti', wm, 'left', left, nl)
