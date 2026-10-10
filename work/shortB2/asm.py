"""asm.py plan.json out.json seed type_n.json type_t.json type_i.json type_c.json -> assemble+add, save if better"""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/shortB2")
from fixB import TypePack, assemble, with_add, save_if_better, check, INST
plan = json.load(open(sys.argv[1])); out = sys.argv[2]; seed = int(sys.argv[3])
tps = {}
for f in sys.argv[4:]:
    d = json.load(open(f)); t = d["t"]
    tp = TypePack(plan, t, seed, log=lambda s: None)
    jidx = {it[1]: j for j, it in enumerate(tp.items)}
    for b in range(len(tp.bins)):
        k = tp.bins[b][0]
        ids = d["asg"].get(str(k))
        if ids:
            tp.set_bin(b, [jidx[i] for i in ids])
        else:
            tp.failed.add(b)
    tps[t] = tp
    print(t, tp.stats(), flush=True)
sol, added = with_add(plan, tps, seed)
sc = save_if_better(sol, print)
json.dump(sol, open(out, "w"))
print("score", sc, "added", added, check(sol)[0])
