import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortB2")
from dpack import initial_d, TypePack
plan = json.load(open(sys.argv[1]))
for t in sys.argv[2]:
    for order in ("asc", "desc"):
        tp = TypePack(plan, t, 0, log=lambda s: None)
        initial_d(tp, order=order)
        print(t, order, 'failedF', [tp.bins[b][1] for b in tp.failed if tp.bins[b][3]][:5], 'nF', sum(1 for b in tp.failed if tp.bins[b][3]), flush=True)
