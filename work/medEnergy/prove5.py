"""Targeted plan search for 5_energy on e5/relax model: LP -> exact mode devs -> fix models with a single mode
within budget U-T, budget row, dual-implied row bounds, MIP with cutoff T. Saves improving plans.
usage: prove5.py T tl mf mt out.json [prune] [seed]"""
import sys, json, time, os
import numpy as np, highspy
from fractions import Fraction
from scipy.sparse import coo_matrix
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/e5")
from relax import build, to_highs, INF
from modedev import mode_devs
from rmip5 import set_mode
T = int(sys.argv[1]); tl = float(sys.argv[2]); mf = int(sys.argv[3]); mt = int(sys.argv[4]); out = sys.argv[5]
prune = int(sys.argv[6]) if len(sys.argv) > 6 else 100; seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
taus = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)
inst = json.load(open("/home/user/EID/datasets/5_energy.json")); M = inst["models"]
t0 = time.time()
mdl = build(inst, taus)
for r, nm in enumerate(mdl.rnames):
    if nm.startswith("free"): mdl.rhi[r] -= mf
    if nm.startswith("tot"): mdl.rhi[r] -= mt
h, _ = to_highs(mdl, integer=False); h.run()
sol = h.getSolution(); y = np.array(sol.row_dual); act = np.array(sol.row_value)
U, modes, red = mode_devs(mdl, y); sgn = 1
if U > 1e9: U, modes, red = mode_devs(mdl, -y); sgn = -1
bud = float(U) - T + 1e-6
print(f"LP {h.getInfo().objective_function_value:.3f} U {float(U):.3f} budget {bud:.3f} ({time.time()-t0:.0f}s)", flush=True)
lo, up = list(mdl.lo), list(mdl.up)
nfree = 0; bestsum = 0.0
def modered(i, md):
    if md == "off": return Fraction(0)
    if md == "full": return red[mdl.fv[i]]
    if md == "src": return red[mdl.sv[i]]
    return red[mdl.hv[i]] + sum(red[mdl.zv[(i, t)]] for t in md[4:])
for i, d in modes.items():
    bm = min(d, key=d.get); bestsum += float(modered(i, bm))
    if sum(1 for v in d.values() if float(v) <= bud) > 1:
        nfree += 1; continue
    set_mode(mdl, lo, up, i, bm)
print("free models", nfree, flush=True)
# Y as in mode_devs
nR = len(mdl.rlo); rlo = list(mdl.rlo); rhi = list(mdl.rhi); nimp = 0
for r in range(nR):
    nm = mdl.rnames[r]
    if nm.startswith("one") or nm.startswith("zh"): continue
    yr = sgn * float(y[r])
    if mdl.rhi[r] == INF and yr > 0: continue
    if mdl.rlo[r] == -INF and yr < 0: continue
    if yr > 1e-12: rlo[r] = max(rlo[r], mdl.rhi[r] - bud / yr - 1e-6); nimp += 1
    elif yr < -1e-12: rhi[r] = min(rhi[r], mdl.rlo[r] + bud / (-yr) + 1e-6); nimp += 1
print("implied row bounds", nimp, flush=True)
# prune tangent rows far from LP active piece
keep = np.ones(nR, bool); groups = {}
for r, nm in enumerate(mdl.rnames):
    if (nm.startswith("tF") or nm.startswith("t")) and "_" in nm and not nm.startswith("tot"):
        g, j = nm.rsplit("_", 1); groups.setdefault(g, []).append((int(j), r))
for g, lst in groups.items():
    sl = [(act[r] - mdl.rlo[r], j) for j, r in lst]; jb = min(sl)[1]
    for j, r in lst:
        if abs(j - jb) > prune: keep[r] = False
modelvars = list(mdl.fv.values()) + list(mdl.hv.values()) + list(mdl.sv.values()) + list(mdl.zv.values())
R = np.array(mdl.rows); C = np.array(mdl.cols); V = np.array(mdl.vals, float)
rm = -np.ones(nR, int); rm[keep] = np.arange(keep.sum()); sel = keep[R]
R2 = list(rm[R[sel]]); C2 = list(C[sel]); V2 = list(V[sel])
rlo2 = [v for v, k in zip(rlo, keep) if k]; rhi2 = [v for v, k in zip(rhi, keep) if k]
nr = len(rlo2)
for j in modelvars:
    R2.append(nr); C2.append(j); V2.append(-float(red[j]))
rlo2.append(-INF); rhi2.append(bud - bestsum + 1e-6); nr += 1
n = len(mdl.c)
A = coo_matrix((V2, (R2, C2)), shape=(nr, n)).tocsc()
hh = highspy.Highs(); hh.setOptionValue("output_flag", True); hh.setOptionValue("log_to_console", False)
hh.setOptionValue("log_file", out + ".highs.log"); hh.setOptionValue("threads", 1)
lp = highspy.HighsLp(); lp.num_col_ = n; lp.num_row_ = nr
lp.col_cost_ = -np.array(mdl.c, float); lp.col_lower_ = np.array(lo, float); lp.col_upper_ = np.array(up, float)
lp.row_lower_ = np.array(rlo2, float); lp.row_upper_ = np.array(rhi2, float)
lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
lp.integrality_ = [highspy.HighsVarType.kInteger if x else highspy.HighsVarType.kContinuous for x in mdl.integ]
hh.passModel(lp)
hh.setOptionValue("time_limit", tl); hh.setOptionValue("mip_rel_gap", 0.0); hh.setOptionValue("mip_abs_gap", 0.5)
hh.setOptionValue("random_seed", seed); hh.setOptionValue("objective_bound", -(T - 0.5))
sols = []
def mkplan(x):
    plan = {"full": [], "half": [], "source": [], "srcreq": []}
    for i, j in mdl.fv.items():
        if x[j] > 0.5: plan["full"].append(i)
    for i, j in mdl.hv.items():
        if x[j] > 0.5: plan["half"].append(i)
    for i, j in mdl.sv.items():
        if x[j] > 0.5: plan["source"].append(i)
    for (i, t), j in mdl.zv.items():
        if x[j] > 0.5: plan["srcreq"].append([i, t])
    plan["obj"] = int(round(sum(mdl.c[j] * round(x[j]) for j in modelvars)))
    return plan
def cb(e):
    x = np.array(e.data_out.mip_solution); p = mkplan(x); p["t"] = time.time() - t0
    sols.append(p); json.dump(sols, open(out, "w")); print("improving", p["obj"], "at", round(p["t"]), flush=True)
hh.cbMipImprovingSolution.subscribe(cb)
hh.run(); info = hh.getInfo()
print("FINAL", hh.modelStatusToString(hh.getModelStatus()), "dual bound", -info.mip_dual_bound, "time", round(time.time() - t0), flush=True)
