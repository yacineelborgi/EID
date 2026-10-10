import json, sys, numpy as np
sys.path.insert(0, '.')
import mip6
inst = json.load(open("../datasets/6_shortage.json"))
mo = mip6.build(inst)
h = mip6.to_highs(mo, integer=False); h.run()
sol=h.getSolution()
dual = np.array(sol.row_dual); act=np.array(sol.row_value)
for i,nm in enumerate(mo.names):
    if nm.startswith("cut_") or nm.startswith("cap_") or nm.startswith("P_"):
        if abs(dual[i])>1e-9 or nm.startswith("cap_"):
            print(nm, "dual", dual[i], "act", act[i]/mip6.SC, "rhi", mo.rhi[i]/mip6.SC)
