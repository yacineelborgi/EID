import numpy as np
TYPES="ntic"
def classify(inst, sol):
    M, D = inst["models"], inst["datasets"]
    dm = {d: m for d, m in sol["dataMappings"]}
    mm = sol["modelMappings"]
    us = {s for s, _ in mm}
    tg = {t for _, t in mm}
    trained = (set(dm.values()) | tg) - us
    cop = {m for d, m in dm.items() if D[d]["isCopyrighted"]}
    full = {i for i in trained if i not in cop and i not in tg}
    half = trained - full
    return dm, mm, full, half, us

def encode(mo, inst, sol):
    """incumbent -> full column vector of mo (K, W, P computed exactly)."""
    M = inst["models"]
    dm, mm, full, half, us = classify(inst, sol)
    srcd = {(tg, next(iter(M[s]["dataRequirements"]))) for s, tg in mm}
    x = np.zeros(mo.nTot)
    for i in full: x[mo.fo + i] = 1
    for i in half: x[mo.ho + i] = 1
    for i in us: x[mo.so + i] = 1
    for k, (i, t, L, isb) in enumerate(mo.zlist):
        if (i, t) in srcd: x[mo.zo + k] = 1
    # continuous: solve rows K/P equalities directly
    A = mo.A.tocsr()
    names = mo.names
    for pool, t, th, K, W in mo.fam:
        if th is None:
            r = names.index(f"P_{pool}_{t}")
            row = A.getrow(r)
            val = -sum(v * x[j] for j, v in zip(row.indices, row.data) if j != K)
            x[K] = val
        else:
            r = names.index(f"K_{pool}_{t}_{th}")
            row = A.getrow(r)
            k = -sum(v * x[j] for j, v in zip(row.indices, row.data) if j != K)
            k = int(round(k))
            x[K] = k
            its = mo.fs[t] if pool == "F" else mo.al[t]
            x[W] = sum(its[:k]) * 1e-4
    return x


def encode_plan(mo, inst, plan):
    full, half, us = set(plan['full']), set(plan['half']), set(plan['source'])
    srcd = {tuple(x) for x in plan['srcreq']}
    x = np.zeros(mo.nTot)
    for i in full: x[mo.fo + i] = 1
    for i in half: x[mo.ho + i] = 1
    for i in us: x[mo.so + i] = 1
    for k, (i, t, L, isb) in enumerate(mo.zlist):
        if (i, t) in srcd: x[mo.zo + k] = 1
    A = mo.A.tocsr()
    names = mo.names
    for pool, t, th, K, W in mo.fam:
        if th is None:
            row = A.getrow(names.index(f"P_{pool}_{t}"))
            x[K] = -sum(v * x[j] for j, v in zip(row.indices, row.data) if j != K)
        else:
            row = A.getrow(names.index(f"K_{pool}_{t}_{th}"))
            k = int(round(-sum(v * x[j] for j, v in zip(row.indices, row.data) if j != K)))
            x[K] = k
            its = mo.fs[t] if pool == "F" else mo.al[t]
            x[W] = sum(its[:k]) * 1e-4
    return x
