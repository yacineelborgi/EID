"""Ruin & recreate LNS for 6_shortage-like instances.
ruin: drop random (source, mono target) pairs, mono data models and low-ratio multi-type
      data models of one or two types (their datasets go back to the pool, energy freed)
recreate: compaction of touched types -> residual CP-SAT addition -> role relabeling MIP.
Keeps the best valid solution.
usage: python lns_rr.py inst sol_in out iters seed"""
import json, sys, random, copy, collections, time
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from sol6 import State, TYPES
import subprocess, os, tempfile
import compact, resid_add  # roles (highspy) runs in a subprocess: highspy and ortools clash
from solve import strict_ok


LP_MULTI = [9201, 13322, 8556, 935, 2930, 8223]
try:
    import numpy as _np
    _lp = json.load(open("lp6_pos.json"))
    LP_POS_MULTI, LP_POS_MONO = _lp["multi"], _lp["mono"]
except (OSError, ValueError):
    LP_POS_MULTI, LP_POS_MONO = [], []


def clone(st):
    n = object.__new__(State)
    n.inst, n.M, n.D = st.inst, st.M, st.D
    n.src = dict(st.src)
    n.tgt = collections.defaultdict(list, {k: list(v) for k, v in st.tgt.items()})
    n.item_bin = dict(st.item_bin)
    n.bin_items = collections.defaultdict(set, {k: set(v) for k, v in st.bin_items.items()})
    n.used = set(st.used)
    return n


def drop(st, m):
    M = st.M
    for t in M[m]["dataRequirements"]:
        for d in list(st.bin_items.get((m, t), ())):
            del st.item_bin[d]
        st.bin_items[(m, t)] = set()
    st.used.discard(m)
    if m in st.src:
        tg = st.src.pop(m)
        st.tgt[tg].remove(m)
    for s in list(st.tgt.get(m, [])):
        # target dropped: its sources become spare -> drop them too
        del st.src[s]
        drop(st, s)
    st.tgt.pop(m, None)


def ruin(st, rng, types, k_pair, k_mono, k_multi, mode="mixed"):
    M = st.M
    mono_t = lambda m: next(iter(M[m]["dataRequirements"])) if len(M[m]["dataRequirements"]) == 1 else None
    victims = []
    if mode == "full":
        fulls = [m for m in st.used if m not in st.src and not st.tgt.get(m) and any(t in M[m]["dataRequirements"] for t in types)]
        victims += rng.sample(fulls, min(len(fulls), rng.randint(5, 30)))
        k_pair = k_mono = k_multi = 0
    elif rng.random() < 0.5:
        # targets whose sources burn the most energy, and the lowest value-per-energy targets
        tg_cost = collections.defaultdict(float)
        for s, tg in st.src.items():
            if any(mono_t(s) == t for t in types):
                tg_cost[tg] += M[s]["energyCost"]
        tgs = sorted(tg_cost, key=lambda m: -(tg_cost[m] + M[m]["energyCost"]) / (M[m]["value"] // 2 + 1))[:60]
        victims += rng.sample(tgs, min(len(tgs), rng.randint(3, 15)))
        k_pair = 0
    for t in types:
        pairs = [(s, tg) for s, tg in st.src.items() if mono_t(s) == t and mono_t(tg) == t and len(st.tgt[tg]) == 1]
        for s, tg in rng.sample(pairs, min(k_pair, len(pairs))):
            victims.append(tg)
        monos = [m for m in st.used if m not in st.src and not st.tgt.get(m) and mono_t(m) == t]
        victims += rng.sample(monos, min(k_mono, len(monos)))
        multis = [m for m in st.used if m not in st.src and not st.tgt.get(m) and mono_t(m) is None and t in M[m]["dataRequirements"]]
        multis.sort(key=lambda m: M[m]["value"] / (M[m]["energyCost"] + 1e9))
        cand = multis[:max(40, 4 * k_multi)]
        victims += rng.sample(cand, min(k_multi, len(cand)))
    touched = set(types)
    dropped = set()
    hint = {}
    for m in set(victims):
        if m in st.used:
            touched |= set(M[m]["dataRequirements"])
            before = set(st.used)
            snap = {}
            for mm in [m] + list(st.tgt.get(m, [])):
                kind = "src" if mm in st.src else ("tgt" if st.tgt.get(mm) else ("full" if st.is_full(mm) else "half"))
                snap[mm] = (kind, {t: list(st.bin_items.get((mm, t), ())) for t in M[mm]["dataRequirements"]})
            drop(st, m)
            dropped |= before - st.used
            hint.update({k: v for k, v in snap.items() if k not in st.used})
    st._hint = hint
    return touched, dropped


def main():
    inst = json.load(open(sys.argv[1]))
    best = State(inst, json.load(open(sys.argv[2])))
    out = sys.argv[3]
    iters, seed = int(sys.argv[4]), int(sys.argv[5])
    rng = random.Random(seed)
    tmpdir = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out)))
    best_sc = best.score()[0]
    M = best.M
    multi_pool = sorted((m for m in range(len(M)) if m not in best.used and len(M[m]["dataRequirements"]) > 1
                         and all(q["upperBound"] >= q["lowerBound"] for q in M[m]["dataRequirements"].values())),
                        key=lambda m: -M[m]["value"] / (M[m]["energyCost"] * 2.8e-10 + 0.0052 * sum(q["lowerBound"] for q in M[m]["dataRequirements"].values()) + 1))[:400]
    print("start", best_sc, flush=True)
    shared = os.path.join(os.path.dirname(os.path.abspath(out)), "rr_shared.json")
    shared_sc = shared + ".score"
    for it in range(iters):
        t0 = time.time()
        try:
            ssc = int(open(shared_sc).read())
            if ssc > best_sc:
                cand = State(inst, json.load(open(shared)))
                c_sc, c_ok, _ = cand.score()
                if c_ok and c_sc > best_sc:
                    best, best_sc = cand, c_sc
                    print(f"it {it} loaded shared best {best_sc}", flush=True)
        except (OSError, ValueError):
            pass
        st = clone(best)
        big = len(sys.argv) > 6 and sys.argv[6] == "big"
        types = list(TYPES) if big else rng.sample(TYPES, rng.choice([1, 1, 2]))
        mode = "full" if (big or rng.random() < 0.5) else "mixed"
        touched, dropped = ruin(st, rng, types, k_pair=rng.randint(2, 10), k_mono=rng.randint(0, 6), k_multi=rng.randint(0, 4), mode=mode)
        # release the items of some kept bins of the touched types: they must be refilled exactly
        refill, refill_hint = [], []
        cov = st.covered()
        kept_bins = [(m, t) for m in st.used if m not in st.src for t in st.M[m]["dataRequirements"]
                     if t in touched and (m, t) not in cov and st.bin_items.get((m, t))]
        nref = rng.randint(150, 300) if big else rng.randint(20, 80)
        chosen = set()
        if rng.random() < 0.5:
            # tiny bins and holders of the smallest datasets of the touched types
            src_bins = [(s, next(iter(st.M[s]["dataRequirements"]))) for s in st.src]
            src_bins = [k for k in src_bins if k[1] in touched and st.bin_items.get(k)]
            pool_b = kept_bins + src_bins
            for t in touched:
                small = set(i for _, i in sorted((d["size"], d["id"]) for d in st.D if d["dataType"] == t)[:rng.randint(60, 200)])
                for k in pool_b:
                    if k[1] == t and (st.M[k[0]]["dataRequirements"][t]["lowerBound"] <= 1000 or st.bin_items[k] & small):
                        chosen.add(k)
        chosen |= set(rng.sample(kept_bins, min(len(kept_bins), nref)))
        for (m, t) in chosen:
            fo = st.is_full(m) if m not in st.src else False
            its = list(st.bin_items[(m, t)])
            for i in its:
                del st.item_bin[i]
            st.bin_items[(m, t)] = set()
            refill.append([m, t, fo]); refill_hint.append(its)
        mc = [m for m in dropped if len(st.M[m]["dataRequirements"]) > 1] + LP_MULTI + LP_POS_MULTI + rng.sample(multi_pool, 60)
        em = [m for m in dropped if len(st.M[m]["dataRequirements"]) == 1] + LP_POS_MONO
        fd, tin = tempfile.mkstemp(suffix=".json", dir=tmpdir); os.close(fd)
        tout, tpar = tin[:-5] + "_r.json", tin[:-5] + "_p.json"
        json.dump(st.to_sol(), open(tin, "w"))
        json.dump({"multi_cands": mc, "extra_mono": em, "tlim": 300 if big else 60, "refill": refill, "refill_hint": refill_hint,
                   "hint": {str(k): [v[0], v[1]] for k, v in getattr(st, "_hint", {}).items()}}, open(tpar, "w"))
        subprocess.run([sys.executable, "recreate.py", sys.argv[1], tin, tpar, tout], capture_output=True)
        if os.path.exists(tout):
            st = State(inst, json.load(open(tout)))
            os.remove(tout)
        for f in (tin, tpar):
            os.remove(f)
        for t in touched:
            compact.compact(st, t, rounds=1, seed=rng.randrange(10 ** 6), log=False, pairs=30)
        sc, ok, msg = st.score()
        tag = ""
        if ok and sc == best_sc and strict_ok(inst, st.to_sol()):
            best = st  # accept sideways moves (keeps compaction progress)
            tag = " same"
        if ok and sc > best_sc and strict_ok(inst, st.to_sol()):
            best, best_sc = st, sc
            json.dump(st.to_sol(), open(out, "w"))
            tag = " NEW BEST"
            try:
                ssc = int(open(shared_sc).read())
            except (OSError, ValueError):
                ssc = 0
            if sc > ssc:
                json.dump(st.to_sol(), open(shared + ".tmp", "w")); os.replace(shared + ".tmp", shared)
                open(shared_sc + ".tmp", "w").write(str(sc)); os.replace(shared_sc + ".tmp", shared_sc)
        print(f"it {it} {mode} types {''.join(types)} touched {''.join(sorted(touched))} -> {sc} ok={ok} best {best_sc} {time.time()-t0:.0f}s{tag}", flush=True)


if __name__ == "__main__":
    main()
