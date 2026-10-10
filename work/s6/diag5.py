import json, sys, random
sys.path.insert(0, '/home/user/EID'); sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import pack6
inst = json.load(open('/home/user/EID/datasets/6_shortage.json'))
plan = json.load(open(sys.argv[1]))
for g0 in [int(x) for x in sys.argv[2:]]:
    pack6.G0 = g0
    dm, failed, free, copy = pack6.assign6(inst, plan, random.Random(0), 0.0)
    print(g0, 'failed', len(failed), 'left', {t: (len(free[t].s), len(copy[t].s)) for t in 'ntic'}, flush=True)
