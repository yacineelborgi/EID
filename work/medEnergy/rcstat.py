import pickle, json, sys
import numpy as np
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
C=pickle.load(open('../med/lpm_fine.npz.cert.pkl','rb'))
B=C['B']; rcs=C['rcs']; print('B',float(B), len(rcs))
inst=json.load(open('/home/user/EID/datasets/2_medium.json')); nM=len(inst['models'])
nZ=sum(len(m['dataRequirements']) for m in inst['models']); nV=3*nM+nZ
r=np.array([float(x) for x in rcs[:nV]])
for T in (272321,272322,272323,272324,272325):
    bud=float(B)-T
    print(T, 'budget',round(bud,3),'free int cols', int((np.abs(r)<=bud).sum()), 'models with free col', len({j% nM for j in np.nonzero(np.abs(r[:3*nM])<=bud)[0]}))
ye=C['ye']; print('nonzero duals', sum(1 for v in ye if v!=0))
