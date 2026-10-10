"""Local plan repair with the joint F/A matching oracle.
usage: locs.py plan.json margin out.json
"""
import json, sys, random
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortA2")
from joint import joint
from ptype import M, INST

CAP = INST["energyCap"]
bad = {i for i, m in enumerate(M) if any(r["upperBound"] < r["lowerBound"] for r in m["dataRequirements"].values())}


def val(p):
    return sum(M[i]["value"] for i in p["full"]) + sum(M[i]["value"] // 2 for i in p["half"])


def energy(p):
    return sum(M[i]["energyCost"] for i in set(p["full"]) | set(p["half"]) | set(p["source"]))


def J(p, ts="ntic"):
    out = {}
    for t in ts:
        best, _ = joint(p, t)
        out[t] = best
    return out


def score(Js, m):
    return min(min(F, A) for _, F, A in Js.values()) - m


def copyp(p):
    return {"full": list(p["full"]), "half": list(p["half"]), "source": list(p["source"]), "srcreq": [list(x) for x in p["srcreq"]]}


def remove(p, i):
    q = copyp(p)
    if i in q["full"]:
        q["full"].remove(i)
    elif i in q["half"]:
        q["half"].remove(i)
        for (j, t) in [tuple(x) for x in q["srcreq"] if x[0] == i]:
            q["srcreq"].remove([j, t])
            cand = [s for s in q["source"] if t in M[s]["dataRequirements"]]
            cand.sort(key=lambda s: -M[s]["dataRequirements"][t]["lowerBound"])
            q["source"].remove(cand[0])
    return q


def demote(p, i):
    q = copyp(p)
    q["full"].remove(i); q["half"].append(i)
    return q


def add(p, i, as_full):
    q = copyp(p)
    (q["full"] if as_full else q["half"]).append(i)
    return q


def main():
    p = json.load(open(sys.argv[1]))
    m = int(sys.argv[2]); out = sys.argv[3]
    p = copyp(p)
    Js = J(p)
    print("start value", val(p), "eslack", CAP - energy(p), Js, flush=True)
    # repair: greedy best ratio
    while score(Js, m) < 0:
        def_t = {t for t, (_, F, A) in Js.items() if min(F, A) < m}
        cands = []
        for i in p["full"] + p["half"]:
            rq = M[i]["dataRequirements"]
            if not any(t in rq for t in def_t):
                continue
            sr = {tuple(x) for x in p["srcreq"]}
            gain = sum(rq[t]["lowerBound"] for t in def_t if t in rq and (i, t) not in sr)
            loss = M[i]["value"] if i in p["full"] else M[i]["value"] // 2
            cands.append((loss / max(gain, 1), "rm", i, loss))
            if i in p["full"]:
                lossd = M[i]["value"] - M[i]["value"] // 2
                gF = sum(rq[t]["lowerBound"] for t in def_t if t in rq)
                cands.append((lossd / max(gF, 1) * 2, "dm", i, lossd))
        cands.sort()
        best = None
        for r, kind, i, loss in cands[:40]:
            q = remove(p, i) if kind == "rm" else demote(p, i)
            ts = "".join(t for t in "ntic" if t in M[i]["dataRequirements"] or kind == "rm")
            J2 = dict(Js); J2.update(J(q, ts))
            imp = min(score(J2, m), 0) - min(score(Js, m), 0)
            # improvement in total deficit
            d1 = sum(max(0, m - min(F, A)) for _, F, A in Js.values())
            d2 = sum(max(0, m - min(F, A)) for _, F, A in J2.values())
            if d2 < d1:
                rr = loss / (d1 - d2)
                if best is None or rr < best[0]:
                    best = (rr, kind, i, loss, q, J2)
        if best is None:
            print("no repair move", flush=True); break
        _, kind, i, loss, p, Js = best
        print(f" {kind} {i} loss {loss} -> value {val(p)} J {Js}", flush=True)
    # add phase
    es = CAP - energy(p)
    used = set(p["full"]) | set(p["half"]) | set(p["source"])
    cand = [i for i in range(len(M)) if i not in used and i not in bad and M[i]["energyCost"] <= es]
    cand.sort(key=lambda i: -M[i]["value"]); cand = cand[:int(sys.argv[4]) if len(sys.argv) > 4 else 200]
    print("add candidates", len(cand), "eslack", es, flush=True)
    for i in cand:
        es = CAP - energy(p)
        if M[i]["energyCost"] > es:
            continue
        for as_full in (True, False):
            q = add(p, i, as_full)
            ts = "".join(M[i]["dataRequirements"].keys())
            J2 = dict(Js); J2.update(J(q, ts))
            if score(J2, m) >= 0:
                p, Js = q, J2
                print(f" add {i} full={as_full} v {M[i]['value']} -> value {val(p)}", flush=True)
                break
    p["obj"] = val(p)
    print("FINAL value", val(p), "eslack", CAP - energy(p), Js, flush=True)
    json.dump(p, open(out, "w"))


if __name__ == "__main__":
    main()
