import sys, json, pickle, time
import numpy as np
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
SP='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/'
T=int(sys.argv[1])
inst=json.load(open('/home/user/EID/datasets/2_medium.json'))
TH=tuple(int(x) for x in open(SP+'med/th_fine.txt').read().strip().split(','))
t0=time.time()
C=pickle.load(open(SP+'med/lpm_fine.npz.cert.pkl','rb'))
print('load',time.time()-t0, C.keys(), float(C['B']))
mo=mip6.build(inst,exact=True,thetas=TH); print('build',time.time()-t0)
r=np.array([float(x) for x in C['rcs']]); bud=float(C['B']-T)
nM=mo.nM
for nm,off in (('full',mo.fo),('half',mo.ho),('src',mo.so)):
    rr=r[off:off+nM]; free=(np.abs(rr)<=bud)&(mo.ub[off:off+nM]>mo.lb[off:off+nM]); one=(rr>bud)&(mo.ub[off:off+nM]>0)
    print(nm,'free',free.sum(),'fixed1',one.sum())
fullfree=(r[mo.fo:mo.fo+nM]>=-bud)&(mo.ub[mo.fo:mo.fo+nM]>0)
D=inst['datasets']
for t in 'ntic':
    fs=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
    for th in (1000,2000,3000,5000):
        bins=[(i,L,inst['models'][i]['dataRequirements'][t]['upperBound']) for (i,tt,L,isb) in mo.zlist if tt==t and not isb and L<=th and fullfree[i]]
        ny=sum(sum(1 for s in fs if s<=U) for i,L,U in bins)
        print(t,th,'cand small bins',len(bins),'yvars',ny)
