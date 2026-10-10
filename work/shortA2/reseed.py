import sys
inf, outf, newf = sys.argv[1:4]
lines = open(inf).read().split("\n")
nb, ni = map(int, lines[0].split())
a = open(outf).read().split()
assert len(a) == ni
open(newf, "w").write("\n".join(lines[:1 + nb + ni] + a) + "\n")
