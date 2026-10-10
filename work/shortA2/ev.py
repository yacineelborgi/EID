import sys
inf, outf = sys.argv[1:3]
v = open(inf).read().split("\n")
nb, ni = map(int, v[0].split())
B = [tuple(map(float, x.split())) for x in v[1:1+nb]]
I = [tuple(map(int, x.split())) for x in v[1+nb:1+nb+ni]]
a = [int(x) for x in open(outf).read().split()]
S = [0]*nb
for j,b in enumerate(a):
    if b >= 0: S[b] += I[j][0]
d = [(b, B[b][0]-S[b]) for b in range(nb) if S[b] < B[b][0]]
w = sum(S[b]-B[b][0] for b in range(nb) if S[b] >= B[b][0])
print("failed", len(d), "def", sum(x for _,x in d), "waste", w, "pool", sum(I[j][0] for j in range(ni) if a[j]<0), [(B[b][0],x) for b,x in d][:5])
