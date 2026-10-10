import sys, json
from hyb import *
st = St(json.load(open(OUT)))
res = {tuple(k) for k in json.load(open(sys.argv[1]))}
newF = {k[1] for k in res if k[0] == 'f'}
ysm = {(k[2], D[k[1]]['dataType']): k[1] for k in res if k[0] == 'y'}
zs = {(k[1], k[2]) for k in res if k[0] == 'z'}
for t in "ntic":
    tot = sum(d['size'] for d in D if d['dataType'] == t and not d['isCopyrighted'])
    big = sum(RQ[i][t][0] for i in newF if t in RQ[i] and (i, t) not in ysm)
    ys = sum(D[d]['size'] - RQ[i][tt][0] for (i, tt), d in ysm.items() if tt == t)
    print(t, 'slack', tot - big - ys, 'smallwaste', ys, 'nz', len([1 for z in zs if z[1] == t]))
print('2146', RQ[2146], 2146 in st.F)
