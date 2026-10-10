"""Exact relaxation slacks of a plan (free / total rows for each tau, exact P and G)."""
import json
import sys

inst = json.load(open("/home/user/EID/datasets/5_energy.json"))
M, D = inst["models"], inst["datasets"]
TAUS = (1, 10, 30, 60, 100, 200, 300, 600, 1000, 2000, 3000, 6000)


def G(Fs, Cs, kf, kh):
    # min over a >= kf of P_F(a) + P_C(kf+kh-a)
    k = kf + kh
    best = None
    PF = sum(Fs[:kf])
    a = kf
    # merge: take kf free first, then the kh smallest of remaining free and copy
    rest = sorted(Fs[kf:kf + kh] + Cs[:kh])
    return PF + sum(rest[:kh])


def check(plan, taus=TAUS, verbose=True):
    full, half, source = set(plan["full"]), set(plan["half"]), set(plan["source"])
    srq = {(i, t) for i, t in plan["srcreq"]}
    res = {}
    for t in "ntic":
        Fs = sorted(d["size"] for d in D if d["dataType"] == t and not d["isCopyrighted"])
        Cs = sorted(d["size"] for d in D if d["dataType"] == t and d["isCopyrighted"])
        lA = [M[i]["dataRequirements"][t]["lowerBound"] for i in full if t in M[i]["dataRequirements"]]
        lB = [M[i]["dataRequirements"][t]["lowerBound"] for i in half | source
              if t in M[i]["dataRequirements"] and (i, t) not in srq]
        sf, st = [], []
        for tau in taus:
            kf = sum(1 for l in lA if l <= tau)
            kh = sum(1 for l in lB if l <= tau)
            free = sum(Fs) - sum(l for l in lA if l > tau) - sum(Fs[:kf])
            tot = sum(Fs) + sum(Cs) - sum(l for l in lA + lB if l > tau) - G(Fs, Cs, kf, kh)
            sf.append(free); st.append(tot)
        res[t] = (min(sf), min(st))
        if verbose:
            print(t, "free slack", min(sf), "tot slack", min(st), "| per tau free", sf[:9], "tot", st[:9])
    return res


if __name__ == "__main__":
    p = json.load(open(sys.argv[1]))
    check(p)
    val = sum(M[i]["value"] for i in p["full"]) + sum(M[i]["value"] // 2 for i in p["half"])
    print("value", val)
