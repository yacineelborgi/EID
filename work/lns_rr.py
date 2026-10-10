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
import compact, resid_add, roles
from solve import strict_ok


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


def ruin(st, rng, types, k_pair, k_mono, k_multi):
    M = st.M
    mono_t = lambda m: next(iter(M[m]["dataRequirements"])) if len(M[m]["dataRequirements"]) == 1 else None
    victims = []
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
    for m in set(victims):
        if m in st.used:
            touched |= set(M[m]["dataRequirements"])
            drop(st, m)
    return touched


def main():
    inst = json.load(open(sys.argv[1]))
    best = State(inst, json.load(open(sys.argv[2])))
    out = sys.argv[3]
    iters, seed = int(sys.argv[4]), int(sys.argv[5])
    rng = random.Random(seed)
    best_sc = best.score()[0]
    print("start", best_sc, flush=True)
    for it in range(iters):
        t0 = time.time()
        st = clone(best)
        types = rng.sample(TYPES, rng.choice([1, 1, 2]))
        touched = ruin(st, rng, types, k_pair=rng.randint(1, 4), k_mono=rng.randint(0, 3), k_multi=rng.randint(0, 2))
        for t in touched:
            compact.compact(st, t, rounds=2, seed=rng.randrange(10 ** 6), log=False, pairs=60)
        resid_add.add_models(st, tlim=60, log=False)
        try:
            r = roles.solve_roles(st, tlim=120)
            if r["obj"] >= r["cur"]:
                roles.apply_roles(st, r)
        except AssertionError:
            pass
        sc, ok, msg = st.score()
        tag = ""
        if ok and sc > best_sc and strict_ok(inst, st.to_sol()):
            best, best_sc = st, sc
            json.dump(st.to_sol(), open(out, "w"))
            tag = " NEW BEST"
        print(f"it {it} types {''.join(types)} touched {''.join(sorted(touched))} -> {sc} ok={ok} best {best_sc} {time.time()-t0:.0f}s{tag}", flush=True)


if __name__ == "__main__":
    main()
