p='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/lsC/real2.py'
s=open(p).read()
old=s[s.index("        bl = [b0] + donors"):s.index("    if failed: return None, failed")]
new='''        bl = [b0] + donors
        its = pl + [d for k in donors for d in asg[k]]
        if it % 3 == 0:
            items = [(D[d]['size'], d, True) for d in its]
            bins = [(k, binfo[k][0], binfo[k][1], True) for k in bl]
            r = mip_assign(bins, items, tl=10)
            if r is None: continue
            _, out, filled, _, _ = r
            for q, k in enumerate(bl): asg[k] = [its[j] for j in out[q]]
        else:
            av = set(its); newf = {}; good = True
            for k in sorted(bl, key=lambda k: -binfo[k][0] * (0.8 + 0.4 * rng.random())):
                rr = dp_min_subset([(D[d]['size'], d) for d in av], binfo[k][0], binfo[k][1])
                if rr is None: good = False; break
                newf[k] = [d for _, d in rr]; av.difference_update(newf[k])
            if not good: continue
            asg.update(newf)
        failed.pop(0)
        log(f"   repaired {b0} type {t} it {it} nd {nd}")
    used = {d for v in asg.values() for d in v}
    log(f"  type {t} leftover {sum(D[d]['size'] for d in freeT if d not in used)} failed {len(failed)}")
'''
s=s.replace(old,new)
open(p,'w').write(s)
