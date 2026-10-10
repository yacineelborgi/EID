"""Min consumption of small F bins (lb<=theta) of a type with free items: exact MIP. usage: smallmin.py plans idx type theta tl"""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/medEnergy")
from fpack import make
from medpack import mip_assign
plans = json.load(open(sys.argv[1])); plans = [plans] if isinstance(plans, dict) else plans
plan = plans[int(sys.argv[2])]; t = sys.argv[3]; th = int(sys.argv[4]); tl = float(sys.argv[5])
tp = make(plan, t, 0)
sb = [b for b in range(len(tp.bins)) if tp.bins[b][1] <= th]
umax = max(tp.bins[b][2] for b in sb)
its = [j for j in range(len(tp.items)) if tp.items[j][0] <= umax]
tot = sum(x[0] for x in tp.items); sumlb = sum(b[1] for b in tp.bins)
print('small bins', len(sb), 'items', len(its), 'slack', tot - sumlb, flush=True)
r = mip_assign([tp.bins[b] for b in sb], [tp.items[j] for j in its], tl=tl, gap=0.0, verbose=False)
import ptype
if r is None: print('no solution', getattr(ptype, 'LAST_BOUND', None)); sys.exit()
st, out, filled, objv, bnd = r
print('status', st, 'min consumption', objv, 'bound', bnd, 'waste', objv - sum(tp.bins[b][1] for b in sb), 'waste LB', bnd - sum(tp.bins[b][1] for b in sb), flush=True)
json.dump({str(tp.bins[sb[q]][0]): [tp.items[its[j]][1] for j in out[q]] for q in range(len(sb))}, open(f'small_{t}.json', 'w'))
