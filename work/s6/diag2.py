import json, sys, random, collections
sys.path.insert(0, '/home/user/EID'); sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import solve2
import pack6
inst = json.load(open('/home/user/EID/datasets/6_shortage.json')); M, D = inst['models'], inst['datasets']
plan = json.load(open(sys.argv[1]))
dm, failed, free, copy = solve2.assign2(inst, plan, random.Random(0), 0.0)
srcreq = {(i, t) for i, t in plan['srcreq']}
full = set(plan['full'])
fill = collections.defaultdict(list)
for d, m in dm.items(): fill[(m, D[d]['dataType'])].append(D[d]['size'])
for t in 'ntic':
    al = sorted(d['size'] for d in D if d['dataType'] == t)
    fsz = sorted(d['size'] for d in D if d['dataType'] == t and not d['isCopyrighted'])
    bins = [(M[i]['dataRequirements'][t]['lowerBound'], i) for i in set(plan['full']) | set(plan['half']) | set(plan['source'])
            if t in M[i]['dataRequirements'] and (i, t) not in srcreq]
    Ls = sorted(L for L, i in bins)
    G = sum(max(a, b) for a, b in zip(Ls, al))
    LF = sorted(L for L, i in bins if i in full)
    GF = sum(max(a, b) for a, b in zip(LF, fsz))
    filled = [(L, i) for L, i in bins if fill[(i, t)]]
    ex_t = sum(sum(fill[(i, t)]) - L for L, i in filled if L < 3000)
    ex_b = sum(sum(fill[(i, t)]) - L for L, i in filled if L >= 3000)
    print(t, 'T', len(al) and sum(al), 'G(plan)', G, 'slack', sum(al) - G, '| F', sum(fsz), 'GF', GF, 'slackF', sum(fsz) - GF,
          '| excess over L: tiny', ex_t, 'big', ex_b, 'sum L tiny', sum(L for L, i in filled if L < 3000))
