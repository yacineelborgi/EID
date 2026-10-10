// Simulated annealing packer for one data type.
// Bins have windows [lb, ub] (fill <= ub is hard), deficit = max(0, lb - fill) is penalised.
// Bin 0 is the pool (no constraint). Items may only go to bins compatible with them
// (freeonly bins accept free items only).
// Objective: minimise total weighted deficit (weight per bin), tie-break: maximise pool volume.
// input (stdin):
//   nI nB iters seed T0 T1
//   nI lines: size isfree bin0   (bin0: initial bin index, 0 = pool)
//   nB lines: lb ub freeonly weight   (bin 0 line is ignored but must be present)
// output: nI lines: bin index; then a line "deficit <D> pool <P>"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

static uint64_t rs = 88172645463325252ULL;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline double urand(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }

int nI, nB;
int *sz, *isfree, *bin, *lb, *ub, *fo, *fill;
double *w; double KFIX = 0;
// items per bin: dense arrays with positions
int **bitems, *bcnt, *bcap, *pos;
int *order; // items sorted by size
int *rank_;

static void badd(int b, int i) {
    if (bcnt[b] == bcap[b]) { bcap[b] = bcap[b] * 2 + 4; bitems[b] = realloc(bitems[b], sizeof(int) * bcap[b]); }
    pos[i] = bcnt[b]; bitems[b][bcnt[b]++] = i; bin[i] = b; fill[b] += sz[i];
}
static void brem(int i) {
    int b = bin[i], p = pos[i], last = bitems[b][--bcnt[b]];
    bitems[b][p] = last; pos[last] = p; fill[b] -= sz[i]; bin[i] = -1;
}
static inline double defc(int b, int f) {
    if (b == 0) return 0.0;
    int d = lb[b] - f; return d > 0 ? w[b] * d + KFIX : 0.0;
}
static inline int compat(int i, int b) { return b == 0 || !fo[b] || isfree[i]; }

int main(void) {
    long long iters; unsigned long long seed; double T0, T1;
    double K0, K1;
    if (scanf("%d %d %lld %llu %lf %lf %lf %lf", &nI, &nB, &iters, &seed, &T0, &T1, &K0, &K1) != 8) return 1;
    KFIX = K0;
    rs ^= seed * 2654435761ULL + 1;
    sz = malloc(sizeof(int) * nI); isfree = malloc(sizeof(int) * nI); bin = malloc(sizeof(int) * nI); pos = malloc(sizeof(int) * nI);
    int *b0 = malloc(sizeof(int) * nI);
    for (int i = 0; i < nI; i++) scanf("%d %d %d", &sz[i], &isfree[i], &b0[i]);
    lb = malloc(sizeof(int) * nB); ub = malloc(sizeof(int) * nB); fo = malloc(sizeof(int) * nB); fill = calloc(nB, sizeof(int)); w = malloc(sizeof(double) * nB);
    bitems = calloc(nB, sizeof(int *)); bcnt = calloc(nB, sizeof(int)); bcap = calloc(nB, sizeof(int));
    for (int b = 0; b < nB; b++) scanf("%d %d %d %lf", &lb[b], &ub[b], &fo[b], &w[b]);
    ub[0] = 2000000000; lb[0] = 0; fo[0] = 0;
    for (int i = 0; i < nI; i++) badd(b0[i], i);
    // sorted order
    order = malloc(sizeof(int) * nI); rank_ = malloc(sizeof(int) * nI);
    for (int i = 0; i < nI; i++) order[i] = i;
    // simple insertion sort by size (nI small)
    for (int a = 1; a < nI; a++) { int v = order[a], j = a - 1; while (j >= 0 && sz[order[j]] > sz[v]) { order[j + 1] = order[j]; j--; } order[j + 1] = v; }
    for (int r = 0; r < nI; r++) rank_[order[r]] = r;
    double D = 0; for (int b = 1; b < nB; b++) D += defc(b, fill[b]);
    double bestD = D; int bestPool = fill[0];
    int *best = malloc(sizeof(int) * nI); for (int i = 0; i < nI; i++) best[i] = bin[i];
    for (long long it = 0; it < iters; it++) {
        double T = T0 * pow(T1 / T0, (double)it / iters);
        if ((it & 1023) == 0) {
            double Kn = K0 * pow(K1 / K0, (double)it / iters);
            // recompute D with the new K
            KFIX = Kn; D = 0; for (int b = 1; b < nB; b++) D += defc(b, fill[b]);
            bestD = 1e18; // best tracked only at final K: reset, will be refreshed below
        }
        int mv = rnd() % 100;
        if (mv < 30) {
            // relocate random item to random bin
            int i = rnd() % nI, X = bin[i];
            int Y = (rnd() % 8 == 0) ? 0 : (int)(rnd() % nB);
            if (Y == X || !compat(i, Y) || fill[Y] + sz[i] > ub[Y]) continue;
            double d = defc(X, fill[X] - sz[i]) - defc(X, fill[X]) + defc(Y, fill[Y] + sz[i]) - defc(Y, fill[Y]);
            double dp = (X == 0 ? -sz[i] : 0) + (Y == 0 ? sz[i] : 0);
            double delta = d - 1e-6 * dp;
            if (delta <= 0 || urand() < exp(-delta / T)) { brem(i); badd(Y, i); D += d; }
        } else {
            // swap i with j of nearby or random size
            int i = rnd() % nI, j;
            if (mv < 85) {
                int r = rank_[i] + (int)(rnd() % 61) - 30; if (r < 0 || r >= nI) continue; j = order[r];
            } else j = rnd() % nI;
            int X = bin[i], Y = bin[j];
            if (X == Y || !compat(i, Y) || !compat(j, X)) continue;
            int dx = sz[j] - sz[i];
            if (fill[X] + dx > ub[X] || fill[Y] - dx > ub[Y]) continue;
            double d = defc(X, fill[X] + dx) - defc(X, fill[X]) + defc(Y, fill[Y] - dx) - defc(Y, fill[Y]);
            double dp = (X == 0 ? dx : 0) + (Y == 0 ? -dx : 0);
            double delta = d + 1e-6 * dp;
            if (delta <= 0 || urand() < exp(-delta / T)) { brem(i); brem(j); badd(Y, i); badd(X, j); D += d; }
        }
        if (it > iters * 0.9 && (D < bestD - 1e-9 || (fabs(D - bestD) < 1e-9 && fill[0] > bestPool))) {
            bestD = D; bestPool = fill[0];
            for (int k = 0; k < nI; k++) best[k] = bin[k];
            if (bestD < 1e-9) { /* keep going to grow pool */ }
        }
    }
    for (int i = 0; i < nI; i++) printf("%d\n", best[i]);
    printf("deficit %.3f pool %d\n", bestD, bestPool);
    return 0;
}
