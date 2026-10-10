import json,sys
sys.path.insert(0,'../s6')
inst=json.load(open('/home/user/EID/datasets/6_shortage.json'));M=inst['models'];D=inst['datasets']
from enc import classify
p=json.load(open(sys.argv[1]))
for f in sys.argv[2:]:
    s=json.load(open(f))
    dm,mm,full,half,us=classify(inst,s)
    tr=full|half
    miss=[i for i in p['full']+p['half'] if i not in tr]
    added=[i for i in tr if i not in set(p['full'])|set(p['half'])]
    print(f, 'missing', [(i,'F' if i in p['full'] else 'H', M[i]['value'], {t:r['lowerBound'] for t,r in M[i]['dataRequirements'].items()}) for i in miss])
    print('   added', [(i, M[i]['value'], 'F' if i in full else 'H') for i in added])
