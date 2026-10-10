"""Solve the LP relaxation of mip6 with a given threshold set, then certify the bound in exact arithmetic."""
import json, sys, subprocess, time
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
th = tuple(sorted(set(int(x) for x in sys.argv[1].split(','))))
out = sys.argv[2]
inst = json.load(open('/home/user/EID/datasets/6_shortage.json'))
mo = mip6.build(inst, thetas=th)
h = mip6.to_highs(mo, integer=False)
h.run()
print('LP', h.getInfo().objective_function_value, flush=True)
np.savez(out, y=np.array(h.getSolution().row_dual))
