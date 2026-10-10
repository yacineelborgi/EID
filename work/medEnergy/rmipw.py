"""rmip5.run with per-type margins. usage: rmipw.py start.json out.json D tl mfs mts [prune]"""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/e5")
import rmip5, relax
start = json.load(open(sys.argv[1])); out = sys.argv[2]; D = float(sys.argv[3]); tl = float(sys.argv[4])
MFS = [int(x) for x in sys.argv[5].split(',')]; MTS = [int(x) for x in sys.argv[6].split(',')]
prune = int(sys.argv[7]) if len(sys.argv) > 7 else 100
_b = relax.build
def build2(inst, taus, **kw):
    mdl = _b(inst, taus, **kw)
    for r, nm in enumerate(mdl.rnames):
        for k, t in enumerate("ntic"):
            if nm.startswith("free" + t): mdl.rhi[r] -= MFS[k]
            if nm.startswith("tot" + t): mdl.rhi[r] -= MTS[k]
    return mdl
rmip5.build = build2
taus = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)
plan = rmip5.run(taus, D, tl, 0, 0, start=start, prune=prune, verbose=False)
plan['slack'] = {k: v for k, v in plan['slack'].items() if k.endswith('200')}
json.dump(plan, open(out, 'w')); print('obj', plan['obj'], 'bound', plan['bound'], flush=True)
