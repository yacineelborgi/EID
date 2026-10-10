"""min waste of small bins (lb<=TH) of a plan's full models for type t (exact MIP), vs slack."""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA")
from ptype import mip_assign
from state import M, D
plan = json.load(open(sys.argv[1])); t = sys.argv[2]; TH = int(sys.argv[3]); tl = float(sys.argv[4])
F = plan['full']
RQ = lambda i: M[i]['dataRequirements'][t]
bins = [(i, RQ(i)['lowerBound'], RQ(i)['upperBound'], True) for i in F if t in M[i]['dataRequirements'] and RQ(i)['lowerBound'] <= TH]
mu = max(b[2] for b in bins)
items = [(d['size'], d['id'], True) for d in D if d['dataType'] == t and not d['isCopyrighted'] and d['size'] <= mu]
tot = sum(d['size'] for d in D if d['dataType'] == t and not d['isCopyrighted'])
slack = tot - sum(RQ(i)['lowerBound'] for i in F if t in M[i]['dataRequirements'])
r = mip_assign(bins, items, tl=tl)
if r is None: print("no sol"); sys.exit()
st, out, filled, obj, bnd = r
lbs = sum(b[1] for b in bins)
print(f"type {t} nbins {len(bins)} items {len(items)} slack {slack} minwaste {obj - lbs:.0f} bound {bnd - lbs:.1f} status {st}")
json.dump({str(bins[q][0]): [items[j][1] for j in js] for q, js in out.items()}, open(sys.argv[1] + f".small_{t}.json", "w"))
