import json, sys, random
sys.path.insert(0, '/home/user/EID')
from assign import Pool, fill_bin
inst = json.load(open('/home/user/EID/datasets/6_shortage.json')); M, D = inst['models'], inst['datasets']
plan = json.load(open(sys.argv[1])); t = sys.argv[2]
full = set(plan['full']); half = set(plan['half']); source = set(plan['source'])
srcreq = {(i, tt) for i, tt in plan['srcreq']}
free = Pool([]); copy = Pool([])
for d in D:
    if d['dataType'] == t: (copy if d['isCopyrighted'] else free).put((d['size'], d['id']))
bf = sorted([(M[i]['dataRequirements'][t]['lowerBound'], M[i]['dataRequirements'][t]['upperBound'], i) for i in full if t in M[i]['dataRequirements']])
ba = sorted([(M[i]['dataRequirements'][t]['lowerBound'], M[i]['dataRequirements'][t]['upperBound'], i) for i in half | source if t in M[i]['dataRequirements'] and (i, t) not in srcreq])
print('F bins', len(bf), 'A bins', len(ba))
for name, bins, pools in (('F', bf, [free]), ('A', ba, [copy, free])):
    for k, (lb, ub, i) in enumerate(bins):
        res = fill_bin(lb, ub, pools)
        if res is None:
            print(name, 'FAIL idx', k, 'lb', lb, 'ub', ub, 'free left', len(free.s), sum(x[0] for x in free.s), 'copy left', len(copy.s), sum(x[0] for x in copy.s), 'src' if i in source else '')
    print(name, 'done: free left', len(free.s), sum(x[0] for x in free.s), 'copy left', len(copy.s), sum(x[0] for x in copy.s))
