import sys, numpy as np
v=open(sys.argv[1]).read().split("\n"); nb,ni=map(int,v[0].split())
B=[tuple(map(float,x.split())) for x in v[1:1+nb]]; I=[tuple(map(int,x.split())) for x in v[1+nb:1+nb+ni]]
s=np.array(sorted(x[0] for x in I)); print('items',ni,'min',s[:10],'quantiles',np.percentile(s,[10,25,50,75,90]))
w=int(sys.argv[2])
# pair sums
ps=np.sort((s[:,None]+s[None,:])[np.triu_indices(ni,1)])
tot1=tot2=0; c3=0; need3=0
for L,U,F,_ in B:
    n1=np.searchsorted(s,L+w,side='right')-np.searchsorted(s,L)
    n2=np.searchsorted(ps,L+w,side='right')-np.searchsorted(ps,L)
    tot1+=n1; tot2+=n2
    if n1+n2==0: need3+=1
print('w',w,'singles',tot1,'pairs',tot2,'bins with no 1/2 pattern',need3, 'bins L>20000', sum(1 for b in B if b[0]>20000))
