p='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/lsC/real2.py'
s=open(p).read()
s=s.replace('''    used = {d for v in asg.values() for d in v}
    log(f"  type {t} leftover''','''    used = {d for v in asg.values() for d in v}
    ws = sorted(((sum(D[d]['size'] for d in v) - binfo[k][0]), binfo[k][0], len(v)) for k, v in asg.items() if k not in fixb or k[0] in NEWY)
    log(f"  waste>0 (non-fixed) {[w for w in ws if w[0] > 0][-25:]}")
    log(f"  type {t} leftover''')
s=s.replace("def pack_t(st, t, fixed_dm, newbins, seed=0, tmax=60, log=log):","NEWY=set()\ndef pack_t(st, t, fixed_dm, newbins, seed=0, tmax=60, log=log):")
open(p,'w').write(s)
