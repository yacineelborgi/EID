"""Pack every new plan appearing in the given plan files; save best legit to best_med.json and out_max."""
import sys, json, os, time, subprocess
D='/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/med'
files=sys.argv[1].split(','); tend=time.time()+float(sys.argv[2])
seen={}
while time.time()<tend:
    for f in files:
        p=os.path.join(D,f)
        if not os.path.exists(p): continue
        try: plans=json.load(open(p))
        except Exception: continue
        n0=seen.get(f,0)
        for k in range(n0,len(plans)):
            if plans[k]['obj']<=272314: continue
            snap=os.path.join(D,f'snap_{f}_{k}.json'); json.dump(plans[k],open(snap,'w'))
            r=subprocess.run(['python3',D+'/pipeM.py',snap,'0',D+'/best_med.json','2'],capture_output=True,text=True)
            print(f,k,plans[k]['obj'],r.stdout.strip().split('\n')[-4:],flush=True)
            r=subprocess.run(['python3',D+'/saveout.py','2_medium',D+'/best_med.json','272312'],capture_output=True,text=True)
            print('  ',r.stdout.strip(),flush=True)
        seen[f]=len(plans)
    time.sleep(15)
