import json, sys
from locs import *
p=copyp(json.load(open(sys.argv[1]))); out=sys.argv[2]
for mv in sys.argv[3:]:
    k,i=mv[:2],int(mv[2:])
    p=remove(p,i) if k=='rm' else demote(p,i) if k=='dm' else add(p,i,k=='af')
p['obj']=val(p)
print(out,'value',val(p),'eslack',CAP-energy(p),J(p))
json.dump(p,open(out,'w'))
