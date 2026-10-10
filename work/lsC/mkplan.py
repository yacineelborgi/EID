import sys, json
from hyb import *
st = St(json.load(open(OUT)))
res = {tuple(k) for k in json.load(open(sys.argv[1]))}
newF = {k[1] for k in res if k[0] == 'f'}
newH = {k[1] for k in res if k[0] == 'h'} | st.fixedT
plan = {"full": sorted(newF), "half": sorted(newH), "source": sorted(st.src),
        "srcreq": [[t, next(iter(M[s]['dataRequirements']))] for s, t in st.src.items()]}
json.dump(plan, open(sys.argv[2], "w"))
print(len(newF), len(newH))
