"""Rigorous LP bound: exact rational weak duality on the integer-coefficient model.
usage: python certify.py lp.npz  (y from the float model built by mip6.build with default args)"""
import json, sys
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/s6')
import mip6
inst = json.load(open('/home/user/EID/datasets/6_shortage.json'))
cap = inst['energyCap']
y = np.load(sys.argv[1])['y']
mip6.SC = 1.0
TH = tuple(sorted(set(int(x) for x in sys.argv[2].split(",")))) if len(sys.argv) > 2 else mip6.THETAS
mo = mip6.build(inst, exact=True, thetas=TH)
assert mo.A.shape[0] == len(y), (mo.A.shape, len(y))
A = mo.A.tocsc()
# check integrality of coefficients
assert np.all(A.data == np.round(A.data)) and np.all(np.abs(A.data) < 2**52)
sig = []
for nm in mo.names:
    if nm.startswith(('P_', 'cap_', 'cut_', 'tan_')): sig.append(Fr(1, 10000))
    elif nm == 'energy': sig.append(Fr(1, cap))
    else: sig.append(Fr(1))
ye = []
term = Fr(0)
nclip = 0
for r, nm in enumerate(mo.names):
    v = Fr(float(y[r])) * sig[r]
    lo, hi = mo.rlo[r], mo.rhi[r]
    if v > 0 and not np.isfinite(hi): v = Fr(0); nclip += 1
    if v < 0 and not np.isfinite(lo): v = Fr(0); nclip += 1
    ye.append(v)
    if v > 0: term += v * Fr(int(hi))
    elif v < 0: term += v * Fr(int(lo))
# column bounds (finite): binaries [lb,ub]; P,W in [0,capP]; K in [0,n]
lb = [Fr(int(a)) if np.isfinite(a) else None for a in mo.lb]
ub = [Fr(int(a)) if np.isfinite(a) else None for a in mo.ub]
for pool, t, th, K, W in mo.fam:
    its = mo.fs[t] if pool == 'F' else mo.al[t]
    capP = Fr(sum(its))
    if th is None:
        lb[K] = Fr(0); ub[K] = capP
    else:
        lb[W] = Fr(0); ub[W] = capP
colterm = Fr(0)
indptr, ind, dat = A.indptr, A.indices, A.data
for j in range(mo.nTot):
    s = Fr(0)
    for p in range(indptr[j], indptr[j + 1]):
        s += Fr(int(dat[p])) * ye[ind[p]]
    rc = Fr(int(mo.cost[j])) - s
    if rc > 0:
        assert ub[j] is not None, j
        colterm += rc * ub[j]
    elif rc < 0:
        assert lb[j] is not None, j
        colterm += rc * lb[j]
B = term + colterm
print("clipped", nclip, "rigorous LP bound", float(B), "floor", B.numerator // B.denominator)
