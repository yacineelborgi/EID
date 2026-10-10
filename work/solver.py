"""Small MILP backend switch: HiGHS (default) or Gurobi (env SOLVER=gurobi).
milp(cost, A, rlo, rhi, sense="max", tlim=60, threads=1, x0=None, gap=0.0, log=False)
  all variables binary. Returns (x, obj, bound)."""
import os
import numpy as np


def milp(cost, A, rlo, rhi, sense="max", tlim=60, threads=1, x0=None, gap=0.0, log=False):
    backend = os.environ.get("SOLVER", "highs").lower()
    if backend == "gurobi":
        return _gurobi(cost, A, rlo, rhi, sense, tlim, threads, x0, gap, log)
    return _highs(cost, A, rlo, rhi, sense, tlim, threads, x0, gap, log)


def _highs(cost, A, rlo, rhi, sense, tlim, threads, x0, gap, log):
    import highspy
    A = A.tocsc()
    n = len(cost)
    h = highspy.Highs()
    h.setOptionValue("output_flag", log)
    h.setOptionValue("time_limit", float(tlim))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("mip_rel_gap", gap)
    lp = highspy.HighsLp()
    lp.num_col_ = n; lp.num_row_ = A.shape[0]
    lp.col_cost_ = np.asarray(cost, float); lp.col_lower_ = np.zeros(n); lp.col_upper_ = np.ones(n)
    lp.row_lower_ = np.asarray(rlo, float); lp.row_upper_ = np.asarray(rhi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.sense_ = highspy.ObjSense.kMaximize if sense == "max" else highspy.ObjSense.kMinimize
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    h.passModel(lp)
    if x0 is not None:
        s0 = highspy.HighsSolution(); s0.col_value = list(x0); s0.value_valid = True
        h.setSolution(s0)
    h.run()
    x = np.array(h.getSolution().col_value)
    info = h.getInfo()
    return x, info.objective_function_value, info.mip_dual_bound


def _gurobi(cost, A, rlo, rhi, sense, tlim, threads, x0, gap, log):
    import gurobipy as gp
    from gurobipy import GRB
    A = A.tocsr()
    n = len(cost)
    rlo = np.asarray(rlo, float); rhi = np.asarray(rhi, float)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if log else 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.TimeLimit = float(tlim)
    m.Params.Threads = int(threads)
    m.Params.MIPGap = gap
    x = m.addMVar(n, vtype=GRB.BINARY, obj=np.asarray(cost, float))
    m.ModelSense = GRB.MAXIMIZE if sense == "max" else GRB.MINIMIZE
    lo_rows = np.where(np.isfinite(rlo))[0]
    hi_rows = np.where(np.isfinite(rhi))[0]
    eq = set(lo_rows) & set(hi_rows)
    eq_rows = np.array(sorted(r for r in eq if rlo[r] == rhi[r]), dtype=int)
    eqs = set(eq_rows.tolist())
    lo_only = np.array([r for r in lo_rows if r not in eqs], dtype=int)
    hi_only = np.array([r for r in hi_rows if r not in eqs], dtype=int)
    if len(eq_rows):
        m.addMConstr(A[eq_rows], x, "=", rlo[eq_rows])
    if len(lo_only):
        m.addMConstr(A[lo_only], x, ">", rlo[lo_only])
    if len(hi_only):
        m.addMConstr(A[hi_only], x, "<", rhi[hi_only])
    if x0 is not None:
        x.Start = np.asarray(x0, float)
    m.optimize()
    if m.SolCount == 0:
        return np.zeros(n), (-np.inf if sense == "max" else np.inf), m.ObjBound
    return np.array(x.X), m.ObjVal, m.ObjBound
