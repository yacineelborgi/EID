"""Lower bounds on packing waste for a plan (sorted matching), per type."""
import json
import sys

inst = json.load(open("/home/user/EID/datasets/5_energy.json"))
M, D = inst["models"], inst["datasets"]


def bounds(plan):
    full, half, source = set(plan["full"]), set(plan["half"]), set(plan["source"])
    srq = {(i, t) for i, t in plan["srcreq"]}
    out = {}
    for t in "ntic":
        fr = sorted(d["size"] for d in D if d["dataType"] == t and not d["isCopyrighted"])
        cp = sorted(d["size"] for d in D if d["dataType"] == t and d["isCopyrighted"])
        al = sorted(fr + cp)
        lA = sorted(M[i]["dataRequirements"][t]["lowerBound"] for i in full if t in M[i]["dataRequirements"])
        lB = sorted(M[i]["dataRequirements"][t]["lowerBound"] for i in half | source
                    if t in M[i]["dataRequirements"] and (i, t) not in srq)
        wA = sum(max(0, s - l) for l, s in zip(lA, fr))
        lAll = sorted(lA + lB)
        wT = sum(max(0, s - l) for l, s in zip(lAll, al))
        out[t] = dict(F=sum(fr), T=sum(al), LA=sum(lA), LB=sum(lB), wA=wA, wT_union=wT,
                      slackA=sum(fr) - sum(lA) - wA, slackT=sum(al) - sum(lA) - sum(lB) - wT)
    return out


if __name__ == "__main__":
    for t, v in bounds(json.load(open(sys.argv[1]))).items():
        print(t, v)
