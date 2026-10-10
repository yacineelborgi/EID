"""LP of mip6 (multi-threshold) model on an instance; saves duals y to npz. usage: lpm.py inst out.npz [thetas]"""
import sys, json, time
import numpy as np
sys.path.insert(0,'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
inst=json.load(open(sys.argv[1]))
TH=tuple(sorted(set(int(x) for x in sys.argv[3].split(',')))) if len(sys.argv)>3 else mip6.THETAS
t0=time.time()
mo=mip6.build(inst,thetas=TH)
print('built',mo.A.shape,time.time()-t0,flush=True)
h=mip6.to_highs(mo,integer=False)
h.run()
print('LP',h.modelStatusToString(h.getModelStatus()),h.getInfo().objective_function_value,time.time()-t0,flush=True)
sol=h.getSolution()
np.savez(sys.argv[2],y=np.array(sol.row_dual),x=np.array(sol.col_value))
