import sys,json
sys.path.insert(0,'.'); sys.path.insert(0,'/home/user/EID')
from ptype import type_data
from collections import Counter
plan=json.load(open(sys.argv[1])); t=sys.argv[2]; a=[int(x) for x in open(sys.argv[3]).read().split()]
bins,items=type_data(plan,t)
S=[0]*len(bins); n=[0]*len(bins)
for j,b in enumerate(a):
    if b>=0: S[b]+=items[j][0]; n[b]+=1
wb=Counter(); cb=Counter()
for b,(k,L,U,F) in enumerate(bins):
    if S[b]>=L:
        key=(0 if L<=1 else 1 if L<=100 else 2 if L<=1000 else 3 if L<=3000 else 4 if L<=10000 else 5)
        wb[key]+=S[b]-L; cb[key]+=1
print('waste by class',sorted(wb.items()),sorted(cb.items()))
print('pool',[items[j][0] for j in range(len(items)) if a[j]<0])
sz=sorted(s for s,_,_ in items); print('smallest items',sz[:40])
Ls=sorted(b[1] for b in bins); print('smallest L',Ls[:90:3])
print('items per bin',sorted(Counter(n).items()))
