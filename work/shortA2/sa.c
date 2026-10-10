// SA for per-type bin covering with upper bounds and free-only bins.
// input file: nb ni \n nb lines: L U F W \n ni lines: s free \n ni lines: assign(-1 pool)
// args: in out seconds seed T0 T1 eps
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
typedef unsigned long long u64;
static u64 rs;
static inline u64 rnd(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return rs; }
static inline double urand(){ return (rnd()>>11)*(1.0/9007199254740992.0); }
int nb, ni; long *L,*U,*S; int *F; double *W; int *sz,*fr,*a,*best;
int *cnt; // items per bin
// items of bin lists
int **bl; int *pos; // pos of item in its bin list (or pool list)
int *pool, npool; double eps;
double P=0; static inline double cb(int b, long s){ return s<L[b] ? (P+(double)(L[b]-s))*W[b] : eps*(double)(s-L[b]); }
int *dlist,*dpos,nd; // deficit bins
static inline void dupd(int b){ int isd = S[b]<L[b]; if(isd && dpos[b]<0){dpos[b]=nd; dlist[nd++]=b;} else if(!isd && dpos[b]>=0){int p=dpos[b]; int last=dlist[--nd]; dlist[p]=last; dpos[last]=p; dpos[b]=-1;} }
static void rem(int j){ int b=a[j]; if(b<0){ int p=pos[j]; int last=pool[--npool]; pool[p]=last; pos[last]=p; } else { int p=pos[j]; int last=bl[b][--cnt[b]]; bl[b][p]=last; pos[last]=p; S[b]-=sz[j]; dupd(b);} a[j]=-2; }
static void add(int j,int b){ a[j]=b; if(b<0){ pool[npool]=j; pos[j]=npool++; } else { bl[b][cnt[b]]=j; pos[j]=cnt[b]++; S[b]+=sz[j]; dupd(b);} }
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"r"); double secs=atof(argv[3]); rs=strtoull(argv[4],0,10)*2654435761ULL+88172645463325252ULL;
  double T0=atof(argv[5]),T1=atof(argv[6]); eps=atof(argv[7]); if(argc>8) P=atof(argv[8]);
  if(fscanf(f,"%d %d",&nb,&ni)!=2) return 1;
  L=malloc(nb*8);U=malloc(nb*8);S=calloc(nb,8);F=malloc(nb*4);W=malloc(nb*8);cnt=calloc(nb,4);
  bl=malloc(nb*sizeof(int*)); for(int b=0;b<nb;b++) bl[b]=malloc(ni*4);
  dlist=malloc(nb*4); dpos=malloc(nb*4); for(int b=0;b<nb;b++) dpos[b]=-1; nd=0;
  for(int b=0;b<nb;b++){ if(fscanf(f,"%ld %ld %d %lf",&L[b],&U[b],&F[b],&W[b])!=4) return 1; }
  sz=malloc(ni*4);fr=malloc(ni*4);a=malloc(ni*4);best=malloc(ni*4);pos=malloc(ni*4);pool=malloc(ni*4);npool=0;
  for(int j=0;j<ni;j++) if(fscanf(f,"%d %d",&sz[j],&fr[j])!=2) return 1;
  for(int b=0;b<nb;b++) dupd(b);
  for(int j=0;j<ni;j++){ int x; if(fscanf(f,"%d",&x)!=1) return 1; add(j,x); }
  fclose(f);
  double cur=0; for(int b=0;b<nb;b++) cur+=cb(b,S[b]);
  double bestf=cur; memcpy(best,a,ni*4);
  long bestdef=0; for(int b=0;b<nb;b++) if(S[b]<L[b]) bestdef+=L[b]-S[b];
  fprintf(stderr,"start f %.1f def %ld nd %d npool %d\n",cur,bestdef,nd,npool);
  clock_t c0=clock(); long it=0; double T=T0; double lr=log(T1/T0);
  while(1){
    if((it&0xFFFF)==0){ double el=(double)(clock()-c0)/CLOCKS_PER_SEC; if(el>secs) break; T=T0*exp(lr*el/secs); }
    it++;
    int mv=rnd()%3;
    if(mv==0){ // relocate random item to random bin/pool or deficit bin
      int j=rnd()%ni; int A=a[j]; int B;
      if(nd>0 && (rnd()&1)) B=dlist[rnd()%nd]; else { B=(int)(rnd()%(nb+1))-1; }
      if(B==A) continue; if(B>=0 && F[B] && !fr[j]) continue; if(B>=0 && S[B]+sz[j]>U[B]) continue;
      double d=0; if(A>=0) d+=cb(A,S[A]-sz[j])-cb(A,S[A]); if(B>=0) d+=cb(B,S[B]+sz[j])-cb(B,S[B]);
      if(d<=0 || urand()<exp(-d/T)){ rem(j); add(j,B); cur+=d; }
    } else { // swap
      int j=rnd()%ni; int k;
      if(mv==2 && nd>0){ int B=dlist[rnd()%nd]; if(cnt[B]==0) continue; k=bl[B][rnd()%cnt[B]]; } else k=rnd()%ni;
      int A=a[j],B=a[k]; if(A==B) continue;
      if(A>=0 && F[A] && !fr[k]) continue; if(B>=0 && F[B] && !fr[j]) continue;
      long dd=sz[k]-sz[j];
      if(A>=0 && S[A]+dd>U[A]) continue; if(B>=0 && S[B]-dd>U[B]) continue;
      double d=0; if(A>=0) d+=cb(A,S[A]+dd)-cb(A,S[A]); if(B>=0) d+=cb(B,S[B]-dd)-cb(B,S[B]);
      if(d<=0 || urand()<exp(-d/T)){ rem(j); rem(k); add(j,B); add(k,A); cur+=d; }
    }
    if(cur<bestf-1e-9){ bestf=cur; memcpy(best,a,ni*4); }
  }
  long def=0,ws=0; for(int b=0;b<nb;b++){ if(S[b]<L[b]) def+=L[b]-S[b]; else ws+=S[b]-L[b]; }
  // best stats
  long *S2=calloc(nb,8); for(int j=0;j<ni;j++) if(best[j]>=0) S2[best[j]]+=sz[j];
  long bdef=0,bws=0; int bnd=0; for(int b=0;b<nb;b++){ if(S2[b]<L[b]){ bdef+=L[b]-S2[b]; bnd++;} else bws+=S2[b]-L[b]; }
  fprintf(stderr,"it %ld end cur %.1f def %ld waste %ld | best f %.1f def %ld nd %d waste %ld\n",it,cur,def,ws,bestf,bdef,bnd,bws);
  FILE*g=fopen(argv[2],"w"); for(int j=0;j<ni;j++) fprintf(g,"%d\n",best[j]); fclose(g);
  return 0;
}
