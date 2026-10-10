import json, sys, random, time
from ptype import TypePack, match_small
from rfill import seq_fill
from sortedcontainers import SortedList
plan=json.load(open(sys.argv[1]))
for t in sys.argv[2]:
  for order in ("desc","asc","rand"):
    for seed in range(2):
      tp=TypePack(plan,t,seed,log=lambda s:None)
      bins,items=tp.bins,tp.items
      small=[b for b in range(len(bins)) if bins[b][1]<=3000]
      r=match_small([bins[b] for b in small],items,lam=float(sys.argv[3]))
      for q,b in enumerate(small):
          if r[q] is not None: tp.set_bin(b,[r[q]])
      rng=random.Random(seed)
      pool=SortedList((items[j][0],j) for j in tp.pool() if items[j][2])
      bF=[(b,bins[b][1],bins[b][2]) for b in range(len(bins)) if bins[b][1]>3000 and bins[b][3]]
      t0=time.time()
      resF,fF=seq_fill(bF,pool,rng,0,int(sys.argv[4]),order)
      leftF=sum(s for s,_ in pool)
      for j in tp.pool():
          if not items[j][2]: pool.add((items[j][0],j))
      for b,its in resF.items():
          for it in its: pass
      bH=[(b,bins[b][1],bins[b][2]) for b in range(len(bins)) if bins[b][1]>3000 and not bins[b][3]]
      resH,fH=seq_fill(bH,pool,rng,0,int(sys.argv[4]),order)
      print(t,order,seed,'F big',len(bF),'failed',len(fF),'failL',sum(x[1] for x in fF),'leftF',leftF,'| H big',len(bH),'failed',len(fH),'failL',sum(x[1] for x in fH),'left',sum(s for s,_ in pool),f'{time.time()-t0:.1f}s',flush=True)
