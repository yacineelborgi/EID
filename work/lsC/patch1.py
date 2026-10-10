p='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/lsC/hyb.py'
s=open(p).read()
a=s
s=s.replace("""            nv(('y', d, i), 0)
        nv(('z', i, t), 0)""","""            nv(('y', d, i), 0)
        if lb > ZMIN: nv(('z', i, t), 0)""")
s=s.replace("""        addrow([(j, 1) for j, s in byb[(i, t)]] + [(col[('z', i, t)], 1), (col[('f', i)], -1)], 0, 0)""","""        addrow([(j, 1) for j, s in byb[(i, t)]] + ([(col[('z', i, t)], 1)] if ('z', i, t) in col else []) + [(col[('f', i)], -1)], 0, 0)""")
s=s.replace("""            ent += [(j, s) for j, s in byb[(i, t)]] + [(col[('z', i, t)], RQ[i][t][0] + pen)]""","""            ent += [(j, s) for j, s in byb[(i, t)]] + ([(col[('z', i, t)], RQ[i][t][0] + pen)] if ('z', i, t) in col else [])""")
s=s.replace("SM = 1000; WCAP = 1200","SM = 1000; WCAP = 1200; ZMIN = 300")
s=s.replace("def solve_h(st, R, P, tl, margin=None, pen=0,","def solve_h(st, R, P, tl, margin=None, pen=5,")
s=s.replace("""            else: x0[col[('z', i, t)]] = 1""","""            elif ('z', i, t) in col: x0[col[('z', i, t)]] = 1""")
print(s.count('ZMIN'))
open(p,'w').write(s)
