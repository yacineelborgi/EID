import sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'/home/user/EID')
from ptype import type_data, match_small
plan=json.load(open(sys.argv[1])); t=sys.argv[2]; th=int(sys.argv[3]); lam=float(sys.argv[4]); out=sys.argv[5]
bins,items=type_data(plan,t)
small=[q for q,b in enumerate(bins) if b[1]<=th]; big=[q for q,b in enumerate(bins) if b[1]>th]
r=match_small([bins[q] for q in small],items,lam=lam)
used={j for j in r if j is not None}
sw=sum(items[j][0] for j in used)-sum(bins[q][1] for q in small)
rest=[j for j in range(len(items)) if j not in used]
tot=sum(items[j][0] for j in rest); totF=sum(items[j][0] for j in rest if items[j][2])
bL=sum(bins[q][1] for q in big); bLF=sum(bins[q][1] for q in big if bins[q][3])
print('small waste',sw,'fail',sum(x is None for x in r),'big slack',tot-bL,'bigF slack',totF-bLF)
with open(out,'w') as f:
    f.write(f"{len(big)} {len(rest)}\n")
    for q in big: f.write(f"{bins[q][1]} {bins[q][2]} {int(bins[q][3])} 1.0\n")
    for j in rest: f.write(f"{items[j][0]} {int(items[j][2])}\n")
    for j in rest: f.write("-1\n")
json.dump({'small':small,'r':r,'big':big,'rest':rest},open(out+'.map','w'))
