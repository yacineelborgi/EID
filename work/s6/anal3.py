import json, sys, collections
inst=json.load(open('/home/user/EID/datasets/6_shortage.json'))
M=inst['models'];D=inst['datasets']
sol=json.load(open(sys.argv[1] if len(sys.argv)>1 else '/home/user/EID/solutions/6_shortage_submission.json'))
dm={d:m for d,m in sol['dataMappings']}
mm=sol['modelMappings']
us={s for s,_ in mm}; tg={t for _,t in mm}
trained=(set(dm.values())|tg)-us
cop={m for d,m in dm.items() if D[d]['isCopyrighted']}
full={i for i in trained if i not in cop and i not in tg}
fill=collections.defaultdict(list)
for d,m in dm.items(): fill[(m,D[d]['dataType'])].append(D[d]['size'])
def G(L, items):
    L=sorted(L); s=sorted(items)
    return sum(max(a,b) for a,b in zip(L,s))
for t in 'ntic':
  fs=sorted(d['size'] for d in D if d['dataType']==t and not d['isCopyrighted'])
  al=sorted(d['size'] for d in D if d['dataType']==t)
  rows=[(m,M[m]['dataRequirements'][t]['lowerBound'],sum(v)) for (m,tt),v in fill.items() if tt==t]
  L=[r[1] for r in rows]; LF=[r[1] for r in rows if r[0] in full]
  useF=sum(r[2] for r in rows if r[0] in full)
  print(t,'all: sumL',sum(L),'G',G(L,al),'T',sum(al),'slack',sum(al)-G(L,al),' | full: sumL',sum(LF),'G',G(LF,fs),'use',useF,'F',sum(fs),'slackF',sum(fs)-G(LF,fs))
