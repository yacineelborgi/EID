import json, sys, random, collections
sys.path.insert(0, '/home/user/EID'); sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import solve2
import pack6
inst = json.load(open('/home/user/EID/datasets/6_shortage.json')); M, D = inst['models'], inst['datasets']
plan = json.load(open(sys.argv[1]))
dm, failed, free, copy = solve2.assign2(inst, plan, random.Random(0), 0.0)
srcreq = {(i, t) for i, t in plan['srcreq']}
full = set(plan['full'])
print('failed', len(failed))
fill = collections.defaultdict(int)
for d, m in dm.items(): fill[(m, D[d]['dataType'])] += D[d]['size']
for t in 'ntic':
    def first_fail(i):
        for tt in 'ntic':
            if tt in M[i]['dataRequirements'] and (i, tt) not in srcreq and fill[(i, tt)] == 0: return tt
    fb = [i for i in failed if first_fail(i) == t]
    # bins of failed models that got no data in t: the failing type is the first type with no fill
    ff = [i for i in fb if i in full]
    lf = sum(it[0] for it in free[t].s); lc = sum(it[0] for it in copy[t].s)
    print(t, 'failed bins (unfilled)', len(fb), 'full', len(ff), 'left free', len(free[t].s), lf, 'left copy', len(copy[t].s), lc,
          'lbs F', sorted(M[i]['dataRequirements'][t]['lowerBound'] for i in ff)[:10],
          'lbs A', sorted(M[i]['dataRequirements'][t]['lowerBound'] for i in fb if i not in full)[:10])
