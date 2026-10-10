"""Poll MIP plan files; pack new plans better than current best (pipeX); if packing loses value, MITM-repair
(mitmP, target best+1) and pack the repaired plans. Saves to out_max via saveout."""
import sys, json, os, time, subprocess
D='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med'
sys.path.insert(0,'/home/user/EID')
from test_solution import getSolutionScore
TXT=open('/home/user/EID/datasets/2_medium.json').read()
OUT='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_max/2_medium_submission.json'
def cur():
    s,ok,_=getSolutionScore(open(OUT).read(),TXT,show_details=False); return s if ok else 0
files=sys.argv[1].split(','); tend=time.time()+float(sys.argv[2]); seen={}
def run(cmd,env=None,to=400):
    e=dict(os.environ); e.update(env or {})
    try: return subprocess.run(cmd,capture_output=True,text=True,env=e,timeout=to).stdout
    except subprocess.TimeoutExpired: return 'TIMEOUT'
while time.time()<tend:
    for f in files:
        p=os.path.join(D,f)
        if not os.path.exists(p): continue
        try: plans=json.load(open(p))
        except Exception: continue
        for k in range(seen.get(f,0),len(plans)):
            best=cur()
            if plans[k]['obj']<=best: continue
            snap=os.path.join(D,f'snapw_{f}_{k}.json'); json.dump([plans[k]],open(snap,'w'))
            o=run(['python3',D+'/pipeX.py',snap,'0',D+'/best_w.json','1'])
            print(f,k,plans[k]['obj'],'pack:',[l for l in o.split('\n') if 'score' in l][:1],flush=True)
            print('  ',run(['python3',D+'/saveout.py','2_medium',D+'/best_w.json','272312']).strip()[-80:],flush=True)
            if cur()<plans[k]['obj']:
                o=run(['python3',D+'/mitmP.py',snap,D+f'/mitmw_{k}.json','8','21','3','80','2000','2e10',str(k),str(cur()+1)],env={'PIDX':'0'})
                print('   mitm:',o.strip().split('\n')[-2][:200] if o.strip() else o,flush=True)
                mp=D+f'/mitmw_{k}.json'
                if os.path.exists(mp) and json.load(open(mp)):
                    n=len(json.load(open(mp)))
                    o=run(['python3',D+'/pipeX.py',mp,','.join(str(i) for i in range(min(n,4))),D+'/best_w.json','1'])
                    print('   packed:',[l for l in o.split('\n') if 'wrote' in l],flush=True)
                    print('  ',run(['python3',D+'/saveout.py','2_medium',D+'/best_w.json','272312']).strip()[-80:],flush=True)
        seen[f]=len(plans)
    time.sleep(10)
