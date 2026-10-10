import json, sys, time
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
inst = json.load(open('/home/user/EID/datasets/6_shortage.json'))
for spec in sys.argv[1:]:
    mA, mF = map(int, spec.split(','))
    marg = {}
    for t in 'ntic':
        marg[('A', t)] = mA; marg[('F', t)] = mF
    mo = mip6.build(inst, marg=marg)
    h = mip6.to_highs(mo, integer=False); h.run()
    print(spec, h.getInfo().objective_function_value, flush=True)
