// Exhaustive two-pass meet-in-the-middle over LP deviations.
// Input file: see prep.py. usage: mm input.txt pass(1|2) maxhits
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
typedef long long ll;
int K; int *grp, *side; double *cst; ll *dvv; ll (*dd)[5];
ll r0[5], L[5]; double lam[5]; ll dvneed; double Bud, X;
// groups per side
typedef struct { int n; int it[3]; double mn; } Group;
Group *GS[2]; int NG[2];
// stored entries
typedef struct { ll E; int f[4]; int dv; float cost; int parent; short item; } Ent;
Ent *ent; ll nent, capent;
double capStore; int storeSide;
void push_ent(ll E, int *f, int dv, double cost, int parent, int item) {
  if (nent >= capent) { capent = capent ? capent * 2 : 1 << 20; ent = realloc(ent, capent * sizeof(Ent)); if (!ent) { fprintf(stderr, "oom\n"); exit(1);} }
  Ent *e = &ent[nent++]; e->E = E; memcpy(e->f, f, 16); e->dv = dv; e->cost = (float)cost; e->parent = parent; e->item = (short)item;
}
void enum_store(int s, int g0, double cost, ll E, int *f, int dv, int parent, int item) {
  int me = (int)nent; push_ent(E, f, dv, cost, parent, item);
  Group *G = GS[s];
  for (int g = g0; g < NG[s]; g++) {
    if (G[g].mn > capStore - cost + 1e-12) break;
    for (int q = 0; q < G[g].n; q++) {
      int it = G[g].it[q]; double c = cst[it];
      if (c > capStore - cost + 1e-12) continue;
      int f2[4]; for (int j = 0; j < 4; j++) f2[j] = f[j] + (int)dd[it][j + 1];
      enum_store(s, g + 1, cost + c, E + dd[it][0], f2, dv + (int)dvv[it], me, it);
    }
  }
}
// layered index
typedef struct { ll E; int f[4]; int orig; int pad; } IE;
#define MAXL 64
int NL; double T[MAXL]; IE *lay[MAXL]; ll *off[MAXL]; int dvmin[MAXL], dvmax[MAXL]; ll lsz[MAXL];
int cmp_cost(const void *a, const void *b) { float x = ent[*(int*)a].cost, y = ent[*(int*)b].cost; return x < y ? -1 : x > y; }
int cmp_ie_dvE(const void *a, const void *b) {
  const IE *x = a, *y = b; int dx = ent[x->orig].dv, dy = ent[y->orig].dv;
  if (dx != dy) return dx < dy ? -1 : 1; return x->E < y->E ? -1 : x->E > y->E; }
void build_layers() {
  int *ord = malloc(sizeof(int) * nent); for (ll i = 0; i < nent; i++) ord[i] = i;
  qsort(ord, nent, sizeof(int), cmp_cost);
  NL = 0; double step = 0.25;
  for (double t = step; t < capStore - 1e-9 && NL < MAXL - 1; t += step) T[NL++] = t;
  T[NL++] = capStore + 1e-9;
  ll p = 0;
  for (int k = 0; k < NL; k++) {
    while (p < nent && ent[ord[p]].cost <= T[k] + 1e-6) p++;
    lsz[k] = p; lay[k] = malloc(sizeof(IE) * (p + 1));
    int mn = 1 << 30, mx = -(1 << 30);
    for (ll i = 0; i < p; i++) { Ent *e = &ent[ord[i]]; IE *z = &lay[k][i]; z->E = e->E; memcpy(z->f, e->f, 16); z->orig = ord[i];
      if (e->dv < mn) mn = e->dv; if (e->dv > mx) mx = e->dv; }
    qsort(lay[k], p, sizeof(IE), cmp_ie_dvE);
    dvmin[k] = mn; dvmax[k] = mx; int R = mx - mn + 2; off[k] = calloc(R + 1, sizeof(ll));
    for (ll i = 0; i < p; i++) off[k][ent[lay[k][i].orig].dv - mn + 1]++;
    for (int r = 1; r <= R; r++) off[k][r] += off[k][r - 1];
    fprintf(stderr, "layer %d T %.3f size %lld dv [%d,%d]\n", k, T[k], p, mn, mx);
  }
  free(ord);
}
// streaming
int pass, streamSide; ll nquery = 0, nscan = 0, nhits = 0, maxhits;
int stk[400]; int depth = 0;
FILE *hout;
void report(int orig, ll EY, int *fY, int dvY, double cY) {
  nhits++;
  Ent *e = &ent[orig];
  ll sl[5]; sl[0] = r0[0] - EY - e->E; for (int j = 0; j < 4; j++) sl[j + 1] = r0[j + 1] - fY[j] - e->f[j];
  int dv = dvY + e->dv;
  if (nhits <= maxhits) {
    fprintf(hout, "HIT dv %d cost %.4f slack %lld %lld %lld %lld %lld items", dv, cY + e->cost, sl[0], sl[1], sl[2], sl[3], sl[4]);
    for (int q = 0; q < depth; q++) fprintf(hout, " %d", stk[q]);
    int x = orig; while (x > 0) { fprintf(hout, " %d", ent[x].item); x = ent[x].parent; }
    fprintf(hout, "\n"); fflush(hout);
  }
}
static inline void query(ll EY, int *fY, int dvY, double cY) {
  double rho = Bud - cY; if (rho < -1e-9) return;
  double capq = rho; if (pass == 1 && capq > capStore) capq = capStore;
  int k = 0; while (k < NL - 1 && T[k] < capq - 1e-9) k++;
  nquery++;
  ll Ehi = r0[0] - L[0] - EY; ll Elo = Ehi - (ll)(rho / lam[0]) - 2;
  int fh[4]; double sup = lam[0] * (double)Ehi;
  for (int j = 0; j < 4; j++) { fh[j] = (int)(r0[j + 1] - L[j + 1] - fY[j]); sup += lam[j + 1] * fh[j]; }
  int dlo = (int)(dvneed - dvY), dhi = (int)floor(sup + 1e-6);
  if (dlo < dvmin[k]) dlo = dvmin[k]; if (dhi > dvmax[k]) dhi = dvmax[k];
  IE *A = lay[k]; ll *O = off[k]; int mn = dvmin[k];
  for (int d = dlo; d <= dhi; d++) {
    ll a = O[d - mn], b = O[d - mn + 1];
    if (a >= b) continue;
    // lower bound E >= Elo
    ll lo = a, hi = b;
    while (lo < hi) { ll m = (lo + hi) >> 1; if (A[m].E < Elo) lo = m + 1; else hi = m; }
    for (ll i = lo; i < b && A[i].E <= Ehi; i++) {
      nscan++;
      IE *z = &A[i];
      if (z->f[0] <= fh[0] && z->f[1] <= fh[1] && z->f[2] <= fh[2] && z->f[3] <= fh[3]) report(z->orig, EY, fY, dvY, cY);
    }
  }
}
double t0;
void stream(int s, int g0, double cost, ll E, int *f, int dv) {
  if (pass == 1 || cost > X + 1e-12) query(E, f, dv, cost);
  Group *G = GS[s];
  for (int g = g0; g < NG[s]; g++) {
    if (G[g].mn > Bud - cost + 1e-12) break;
    for (int q = 0; q < G[g].n; q++) {
      int it = G[g].it[q]; double c = cst[it];
      if (c > Bud - cost + 1e-12) continue;
      int f2[4]; for (int j = 0; j < 4; j++) f2[j] = f[j] + (int)dd[it][j + 1];
      stk[depth++] = it;
      stream(s, g + 1, cost + c, E + dd[it][0], f2, dv + (int)dvv[it]);
      depth--;
    }
    if (depth == 0) { fprintf(stderr, "  top group %d/%d queries %lld scans %lld hits %lld t %.1f\n", g, NG[s], nquery, nscan, nhits, (double)clock() / CLOCKS_PER_SEC - t0); }
  }
}
int cmpg(const void *a, const void *b) { double x = ((Group*)a)->mn, y = ((Group*)b)->mn; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
  FILE *fp = fopen(argv[1], "r"); pass = atoi(argv[2]); maxhits = atoll(argv[3]); hout = stdout;
  for (int j = 0; j < 5; j++) fscanf(fp, "%lld", &r0[j]);
  for (int j = 0; j < 5; j++) fscanf(fp, "%lf", &lam[j]);
  for (int j = 0; j < 5; j++) fscanf(fp, "%lld", &L[j]);
  fscanf(fp, "%lld %lf %lf %d", &dvneed, &Bud, &X, &K);
  grp = malloc(K * sizeof(int)); side = malloc(K * sizeof(int)); cst = malloc(K * sizeof(double)); dvv = malloc(K * sizeof(ll)); dd = malloc(K * sizeof(*dd));
  int maxg = 0;
  for (int k = 0; k < K; k++) { fscanf(fp, "%d %d %lf %lld", &grp[k], &side[k], &cst[k], &dvv[k]); for (int j = 0; j < 5; j++) fscanf(fp, "%lld", &dd[k][j]); if (grp[k] > maxg) maxg = grp[k]; }
  for (int s = 0; s < 2; s++) {
    Group *G = calloc(maxg + 1, sizeof(Group)); int *map = malloc((maxg + 1) * sizeof(int)); for (int g = 0; g <= maxg; g++) map[g] = -1;
    int ng = 0;
    for (int k = 0; k < K; k++) if (side[k] == s && cst[k] <= Bud + 1e-9) {
      int g = grp[k]; if (map[g] < 0) { map[g] = ng; G[ng].n = 0; G[ng].mn = 1e9; ng++; }
      Group *gg = &G[map[g]]; gg->it[gg->n++] = k; if (cst[k] < gg->mn) gg->mn = cst[k]; }
    qsort(G, ng, sizeof(Group), cmpg); GS[s] = G; NG[s] = ng; free(map);
    fprintf(stderr, "side %d groups %d\n", s, ng);
  }
  t0 = (double)clock() / CLOCKS_PER_SEC;
  // pass1: store side 0 up to X, stream side 1 up to Bud. pass2: store side 1 up to Bud-X, stream side 0 (cost > X)
  storeSide = pass == 1 ? 0 : 1; streamSide = 1 - storeSide;
  capStore = pass == 1 ? X : Bud - X;
  int f0[4] = {0, 0, 0, 0};
  enum_store(storeSide, 0, 0.0, 0, f0, 0, -1, -1);
  fprintf(stderr, "stored %lld entries (cap %.3f) t %.1f\n", nent, capStore, (double)clock() / CLOCKS_PER_SEC - t0);
  build_layers();
  fprintf(stderr, "layers built t %.1f\n", (double)clock() / CLOCKS_PER_SEC - t0);
  stream(streamSide, 0, 0.0, 0, f0, 0);
  fprintf(stderr, "DONE pass %d queries %lld scans %lld hits %lld t %.1f\n", pass, nquery, nscan, nhits, (double)clock() / CLOCKS_PER_SEC - t0);
  printf("DONE pass %d queries %lld scans %lld hits %lld\n", pass, nquery, nscan, nhits);
  return 0;
}
