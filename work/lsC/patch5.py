p='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/lsC/hyb.py'
s=open(p).read()
old="""            if s > ub or s > lb + WCAP: break
            nv(('y', d, i), 0)"""
new="""            if s > ub or s > lb + WCAP: break
            nv(('y', d, i), 0)
        if i in R:
            cur = [d for d in st.used[i] if D[d]['dataType'] == t]
            if len(cur) == 1 and ('y', cur[0], i) not in col: nv(('y', cur[0], i), 0)"""
assert old in s; s=s.replace(old,new); open(p,'w').write(s)
p='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/lsC/real2.py'
s=open(p).read()
old="""    ysm = {(k[2], D[k[1]]['dataType']): k[1] for k in res if k[0] == 'y'}"""
new="""    ysm = {(k[2], D[k[1]]['dataType']): [k[1]] for k in res if k[0] == 'y'}
    ysm.update({(k[1], k[2]): list(k[3]) for k in res if k[0] == 'p'})"""
assert old in s; s=s.replace(old,new)
old="""        for (i, tt), d in ysm.items():
            if tt == t: fixed[d] = (i, t)"""
new="""        for (i, tt), ds in ysm.items():
            if tt == t:
                for d in ds: fixed[d] = (i, t)"""
assert old in s; s=s.replace(old,new)
s=s.replace("res = {tuple(k) for k in json.load(open(sys.argv[1]))}","res = {tuple(tuple(x) if isinstance(x, list) else x for x in k) for k in json.load(open(sys.argv[1]))}")
open(p,'w').write(s)
p='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/lsC/hdrv.py'
s=open(p).read()
s=s.replace("""    sol, failed = realize(st, R, res)""","""    from real2 import realize2
    sol, failed = realize2(st, R, res, tmax=60)""")
open(p,'w').write(s)
