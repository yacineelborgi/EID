"""Load 7_big into numpy arrays (cached npz)."""
import json, os
import numpy as np
TYPES = "ntic"
P = "/home/user/EID/datasets/7_big.json"
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data7.npz")

def load():
    if os.path.exists(C):
        z = np.load(C)
        return {k: z[k] for k in z.files}
    inst = json.load(open(P))
    M, D = inst["models"], inst["datasets"]
    nM = len(M)
    v = np.array([m["value"] for m in M], np.int64)
    e = np.array([m["energyCost"] for m in M], np.int64)
    lb = np.zeros((nM, 4), np.int64); ub = np.zeros((nM, 4), np.int64); has = np.zeros((nM, 4), bool)
    for i, m in enumerate(M):
        for t, r in m["dataRequirements"].items():
            j = TYPES.index(t); has[i, j] = True; lb[i, j] = r["lowerBound"]; ub[i, j] = r["upperBound"]
    ds = np.array([d["size"] for d in D], np.int64)
    dt = np.array([TYPES.index(d["dataType"]) for d in D], np.int64)
    dc = np.array([d["isCopyrighted"] for d in D], np.int64)
    out = dict(v=v, e=e, lb=lb, ub=ub, has=has, ds=ds, dt=dt, dc=dc, cap=np.array(inst["energyCap"], np.int64))
    np.savez(C, **out)
    return out

if __name__ == "__main__":
    d = load()
    print({k: x.shape for k, x in d.items()})
