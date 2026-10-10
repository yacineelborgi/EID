import sys, json, os
best = {}
for f in sys.argv[1:]:
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); t = d["t"]; st = d["stats"]
    key = (len(d["failed"]), st["failL"] - st["left"])
    if t not in best or key < best[t][0]:
        best[t] = (key, f)
print(" ".join(best[t][1] for t in "ntic" if t in best))
