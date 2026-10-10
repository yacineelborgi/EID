"""Per-type exact packer for 6_shortage plans.

Per type t: bins = (model, lb, ub, isF). F bins (full models) take free items only, H bins (half non-src
requirements + sources) take any item. Goal: fill every bin.
  1. small bins (lb <= theta): MIP min consumption (free consumption capped so that the big F bins
     keep enough free mass);
  2. big bins: exact fills (sum == lb) with exact_assign.Packer, F first then H; the `nfinal` widest
     bins are kept for the end and absorb the remainder;
  3. LNS (MIP over failed bins + donor bins + pool) to fill the failures / reduce waste.
"""
import sys, json, time, random
import numpy as np
import highspy
from scipy.sparse import coo_matrix

sys.path.insert(0, "/home/user/EID")
from sortedcontainers import SortedList
from exact_assign import Packer

TYPES = "ntic"
INST = json.load(open("/home/user/EID/datasets/6_shortage.json"))
M, D = INST["models"], INST["datasets"]


def type_data(plan, t):
    full = set(plan["full"]); half = set(plan["half"]); src = set(plan["source"])
    sr = {tuple(x) for x in plan["srcreq"]}
    items = [(d["size"], d["id"], not d["isCopyrighted"]) for d in D if d["dataType"] == t]
    bins = []
    for i in sorted(full | half | src):
        r = M[i]["dataRequirements"].get(t)
        if r is None or (i, t) in sr:
            continue
        bins.append((i, r["lowerBound"], r["upperBound"], i in full))
    return bins, items


def mip_assign(bins, items, tl=30.0, opt=None, wopt=None, free_cap=None, obj="min", gap=1e-6, verbose=False,
               start=None):
    """bins: list (key, lb, ub, isF); items list (size, id, free).
    opt: set of bin indices that are optional (y var, reward wopt[b]); others required.
    obj 'min': minimise consumption (minus rewards). Returns (status, dict b -> [item idx], filled set) or None."""
    nb, ni = len(bins), len(items)
    opt = opt or set()
    var = []
    for b, (k, L, U, isF) in enumerate(bins):
        for j, (s, _, fr) in enumerate(items):
            if s <= U and (fr or not isF):
                var.append((b, j))
    nx = len(var)
    yb = sorted(opt)
    ny = len(yb)
    nv = nx + ny
    yidx = {b: nx + q for q, b in enumerate(yb)}
    rows, cols, vals = [], [], []
    # rows: for each bin two rows (>= lb*y, <= ub*y) -> use: sum s x - lb y >= 0 ; sum s x - ub y <= 0
    # required bins: single ranged row
    for q, (b, j) in enumerate(var):
        rows.append(b); cols.append(q); vals.append(items[j][0])
        rows.append(nb + b); cols.append(q); vals.append(items[j][0])
        rows.append(2 * nb + j); cols.append(q); vals.append(1)
    rlo = np.zeros(2 * nb + ni); rhi = np.zeros(2 * nb + ni)
    for b, (k, L, U, isF) in enumerate(bins):
        if b in opt:
            rows.append(b); cols.append(yidx[b]); vals.append(-L)
            rows.append(nb + b); cols.append(yidx[b]); vals.append(-U)
            rlo[b] = 0; rhi[b] = np.inf
            rlo[nb + b] = -np.inf; rhi[nb + b] = 0
        else:
            rlo[b] = L; rhi[b] = np.inf
            rlo[nb + b] = -np.inf; rhi[nb + b] = U
    for j in range(ni):
        rlo[2 * nb + j] = 0; rhi[2 * nb + j] = 1
    nr = 2 * nb + ni
    if free_cap is not None:
        for q, (b, j) in enumerate(var):
            if items[j][2]:
                rows.append(nr); cols.append(q); vals.append(items[j][0])
        rlo = np.append(rlo, -np.inf); rhi = np.append(rhi, free_cap); nr += 1
    A = coo_matrix((vals, (rows, cols)), shape=(nr, nv)).tocsc()
    cost = np.zeros(nv)
    for q, (b, j) in enumerate(var):
        cost[q] = items[j][0]
    for b in yb:
        cost[yidx[b]] = -wopt[b]
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", tl)
    h.setOptionValue("mip_rel_gap", gap)
    lp = highspy.HighsLp()
    lp.num_col_ = nv; lp.num_row_ = nr
    lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(nv); lp.col_upper_ = np.ones(nv)
    lp.row_lower_ = rlo; lp.row_upper_ = rhi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * nv
    h.passModel(lp)
    if start is not None:
        # start: dict b -> list of item idx (feasible solution)
        xs = np.zeros(nv)
        pos = {bj: q for q, bj in enumerate(var)}
        for b, js in start.items():
            for j in js:
                xs[pos[(b, j)]] = 1
            if b in yidx and js:
                xs[yidx[b]] = 1
        sol = highspy.HighsSolution()
        sol.col_value = list(xs)
        sol.value_valid = True
        h.setSolution(sol)
    h.run()
    x = h.getSolution().col_value
    if len(x) != nv or h.getInfo().primal_solution_status != 2:
        global LAST_BOUND
        LAST_BOUND = (h.getModelStatus(), h.getInfo().mip_dual_bound)
        return None
    out = {b: [] for b in range(nb)}
    for q, (b, j) in enumerate(var):
        if x[q] > 0.5:
            out[b].append(j)
    filled = set()
    for b, (k, L, U, isF) in enumerate(bins):
        s = sum(items[j][0] for j in out[b])
        if L <= s <= U:
            filled.add(b)
        elif s == 0 and b in opt:
            pass
        else:
            return None
    return h.getModelStatus(), out, filled, h.getInfo().objective_function_value, h.getInfo().mip_dual_bound


def match_small(bins, items, lam=0.0, BIGC=10**9):
    """single item per bin, min total size (free items in H bins cost +lam). Returns list item idx/None."""
    from scipy.optimize import linear_sum_assignment
    sz = np.array([s for s, _, _ in items], float); fr = np.array([f for _, _, f in items])
    C = np.full((len(bins), len(items)), float(BIGC))
    for q, (k, L, U, F) in enumerate(bins):
        ok = (sz >= L) & (sz <= U)
        if F:
            ok &= fr
            C[q, ok] = sz[ok]
        else:
            C[q, ok] = sz[ok] + lam * fr[ok]
    r, c = linear_sum_assignment(C)
    out = [None] * len(bins)
    for q, j in zip(r, c):
        if C[q, j] < BIGC:
            out[q] = int(j)
    return out


class TypePack:
    def __init__(self, plan, t, seed=0, log=print):
        self.t = t
        self.bins, self.items = type_data(plan, t)
        self.rng = random.Random(seed)
        self.log = log
        self.bidx = {b[0]: q for q, b in enumerate(self.bins)}
        self.asg = {}  # bin index -> list of item idx
        self.used = {}  # item idx -> bin index
        self.failed = set()

    def pool(self):
        return [j for j in range(len(self.items)) if j not in self.used]

    def set_bin(self, b, js):
        old = self.asg.pop(b, [])
        for j in old:
            del self.used[j]
        if js:
            self.asg[b] = list(js)
            for j in js:
                assert j not in self.used
                self.used[j] = b

    def stats(self):
        tot = sum(s for s, _, _ in self.items)
        filledL = sum(self.bins[b][1] for b in self.asg)
        cons = sum(self.items[j][0] for j in self.used)
        failL = sum(self.bins[b][1] for b in self.failed)
        return dict(nb=len(self.bins), failed=len(self.failed), failL=failL, waste=cons - filledL, left=tot - cons)

    def initial(self, theta=3000, nfinal=30, tl_small=60, order="lb_desc", K=300, G0=20000, lam=0.0):
        bins, items = self.bins, self.items
        small = [b for b in range(len(bins)) if bins[b][1] <= theta]
        big = [b for b in range(len(bins)) if bins[b][1] > theta]
        r = match_small([bins[b] for b in small], items, lam=lam)
        if r is None:
            self.log(f"  t{self.t}: small matching failed")
            return False
        for q, b in enumerate(small):
            if r[q] is None:
                self.failed.add(b)
            else:
                self.set_bin(b, [r[q]])
        sw = sum(items[j][0] for b in small if b in self.asg for j in self.asg[b]) - sum(bins[b][1] for b in small if b in self.asg)
        self.log(f"  t{self.t}: small {len(small)} waste {sw}")
        # big bins
        fin = set(sorted(big, key=lambda b: -(bins[b][2] - bins[b][1]))[:nfinal])
        rest = [b for b in big if b not in fin]
        if order == "lb_desc":
            rest.sort(key=lambda b: (not bins[b][3], -bins[b][1]))
        else:
            rest.sort(key=lambda b: (not bins[b][3], bins[b][1]))
        # F pool / all pool
        poolF = SortedList((items[j][0], j) for j in self.pool() if items[j][2])
        poolC = SortedList((items[j][0], j) for j in self.pool() if not items[j][2])
        pk = Packer([], self.rng, K=K, G0=G0)
        pk.pool = poolF
        mode = "F"
        for b in rest:
            k, L, U, isF = bins[b]
            if not isF and mode == "F":
                for it in poolC:
                    poolF.add(it)
                mode = "A"
            r = pk.fill(L, U, 0)
            if r is None:
                r = pk.fill(L, U, 50)
            if r is None:
                self.failed.add(b); continue
            self.set_bin(b, [j for _, j in r])
        if mode == "F":
            for it in poolC:
                poolF.add(it)
        self.log(f"  t{self.t}: after big {self.stats()} fin {len(fin)} finL {sum(bins[b][1] for b in fin)}")
        # final bins: MIP over the pool
        finl = sorted(fin)
        pl = self.pool()
        r = mip_assign([bins[b] for b in finl], [items[j] for j in pl], tl=60, opt=set(range(len(finl))),
                       wopt={q: 1e6 for q in range(len(finl))})
        if r is None:
            self.failed |= set(finl)
        else:
            st, out, filled, objv, bnd = r
            for q, b in enumerate(finl):
                if q in filled:
                    self.set_bin(b, [pl[j] for j in out[q]])
                else:
                    self.failed.add(b)
        self.log(f"  t{self.t}: initial done {self.stats()}")
        return True

    def lns_fix(self, b0, ndonor=30, tl=10.0, mode="mixed"):
        """Try to fill failed bin b0 by re-solving it with donors + pool."""
        bins, items = self.bins, self.items
        L0 = bins[b0][1]
        filled = list(self.asg.keys())
        if mode == "near":
            cand = sorted(filled, key=lambda b: abs(bins[b][1] - L0) + self.rng.random() * 3000)[:ndonor]
        elif mode == "waste":
            wb = sorted(filled, key=lambda b: -(sum(items[j][0] for j in self.asg[b]) - bins[b][1]) - self.rng.random())
            cand = wb[:ndonor // 2] + self.rng.sample(filled, ndonor - ndonor // 2)
        else:
            cand = self.rng.sample(filled, min(ndonor, len(filled)))
        cand = list(dict.fromkeys(cand))
        isF0 = bins[b0][3]
        sub = [b0] + cand
        pl = self.pool()
        its = pl + [j for b in cand for j in self.asg[b]]
        start = {}
        for q, b in enumerate(sub):
            if b in self.asg:
                start[q] = [its.index(j) for j in self.asg[b]]
        r = mip_assign([bins[b] for b in sub], [items[j] for j in its], tl=tl)
        if r is None:
            return False
        st, out, filled_, objv, bnd = r
        for q, b in enumerate(sub):
            self.set_bin(b, [])
        for q, b in enumerate(sub):
            self.set_bin(b, [its[j] for j in out[q]])
        self.failed.discard(b0)
        return True


def run_type(plan, t, seed=0, theta=3000, nfinal=30, order="lb_asc", log=print):
    tp = TypePack(plan, t, seed, log)
    tp.initial(theta=theta, nfinal=nfinal, order=order)
    return tp


if __name__ == "__main__":
    plan = json.load(open(sys.argv[1]))
    ts = sys.argv[2] if len(sys.argv) > 2 else TYPES
    theta = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    nfinal = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    for t in ts:
        t0 = time.time()
        tp = run_type(plan, t, theta=theta, nfinal=nfinal, log=lambda s: print(s, flush=True))
        print(t, tp.stats(), "failed bins", [tp.bins[b][:4] for b in tp.failed], f"{time.time()-t0:.0f}s", flush=True)
