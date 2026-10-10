p='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/lsC/hyb.py'
s=open(p).read()
s=s.replace("SM = 1000; WCAP = 1200; ZMIN = 300","SM = 3000; WCAP = 1200; ZMIN = 10**9; PW = 3; NPAIR = 25")
old="""        if lb > ZMIN: nv(('z', i, t), 0)"""
new="""        if lb > ZMIN: nv(('z', i, t), 0)
        pats = set()
        if i in R:
            cur = tuple(sorted(d for d in st.used[i] if D[d]['dataType'] == t))
            if len(cur) > 1: pats.add(cur)
        if lb > 300:
            sz = [(D[d]['size'], d) for d in psorted[t] if D[d]['size'] < lb]
            import bisect
            ss = [a for a, _ in sz]; cand = []
            for q, (s1, d1) in enumerate(sz):
                lo = bisect.bisect_left(ss, lb - s1); hi = bisect.bisect_right(ss, lb + PW - s1)
                for q2 in range(max(lo, q + 1), hi):
                    cand.append((min(s1, ss[q2]), s1 + ss[q2] - lb, d1, sz[q2][1]))
            cand.sort(key=lambda c: (c[1], -c[0]))
            for c in cand[:NPAIR]: pats.add(tuple(sorted((c[2], c[3]))))
        for pt in pats: nv(('p', i, t, pt), 0)"""
assert old in s
s=s.replace(old,new)
old2="""    for name, j in col.items():
        if name[0] == 'y':
            byd[name[1]].append(j); byb[(name[2], D[name[1]]['dataType'])].append((j, D[name[1]]['size']))"""
new2="""    for name, j in col.items():
        if name[0] == 'y':
            byd[name[1]].append(j); byb[(name[2], D[name[1]]['dataType'])].append((j, D[name[1]]['size']))
        elif name[0] == 'p':
            for d in name[3]: byd[d].append(j)
            byb[(name[1], name[2])].append((j, sum(D[d]['size'] for d in name[3])))"""
assert old2 in s
s=s.replace(old2,new2)
old3="""            if len(ds) == 1 and ('y', ds[0], i) in col: x0[col[('y', ds[0], i)]] = 1
            elif"""
new3="""            if len(ds) == 1 and ('y', ds[0], i) in col: x0[col[('y', ds[0], i)]] = 1
            elif ('p', i, t, tuple(sorted(ds))) in col: x0[col[('p', i, t, tuple(sorted(ds)))]] = 1
            elif"""
assert old3 in s
s=s.replace(old3,new3)
open(p,'w').write(s)
