"""Verify (official checker + strict_ok + integer energy) and copy to out_max if strictly better."""
import sys, json, os, subprocess
os.chdir('/home/user/EID'); sys.path.insert(0,'/home/user/EID')
from solve import strict_ok
from test_solution import getSolutionScore
name=sys.argv[1]; f=sys.argv[2]; cur=int(sys.argv[3])
INST=f'/home/user/EID/datasets/{name}.json'; txt=open(INST).read(); inst=json.loads(txt)
sol=json.load(open(f))
sc,ok,msg=getSolutionScore(json.dumps(sol),txt,show_details=False)
r=subprocess.run(['python3','/home/user/EID/test_solution.py','-d',INST,'-s',f],capture_output=True,text=True).stdout
so=strict_ok(inst,sol)
used=set(m for _,m in sol['dataMappings'])|set(x for p in sol['modelMappings'] for x in p)
E=sum(inst['models'][i]['energyCost'] for i in used)
print(r.strip().replace('\n',' | '),'strict_ok',so,'E<=cap',E<=inst['energyCap'])
out=f'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_max/{name}_submission.json'
old=cur
if os.path.exists(out):
    o,ok2,_=getSolutionScore(open(out).read(),txt,show_details=False); old=max(old,o if ok2 else -1)
if ok and so and 'valid!' in r and E<=inst['energyCap'] and sc>old:
    json.dump(sol,open(out,'w')); print('SAVED',out,sc,'(prev',old,')')
else: print('not saved', sc, 'old', old)
