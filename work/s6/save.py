import json, sys, shutil, subprocess
sys.path.insert(0, '/home/user/EID')
from solve import strict_ok, score_of
src = sys.argv[1]
OUT = '/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_max/6_shortage_submission.json'
txt = open('/home/user/EID/datasets/6_shortage.json').read(); inst = json.loads(txt)
sol = json.load(open(src)); sc, ok, msg = score_of(txt, sol)
old = json.load(open(OUT)); osc, ook, _ = score_of(txt, old)
r = subprocess.run(['python3', '/home/user/EID/test_solution.py', '-d', '/home/user/EID/datasets/6_shortage.json', '-s', src], capture_output=True, text=True, cwd='/home/user/EID').stdout
print(r.strip().replace('\n', ' '), 'strict', strict_ok(inst, sol), 'old', osc)
if ok and strict_ok(inst, sol) and 'valid!' in r and sc > osc:
    shutil.copy(src, OUT); print('SAVED', sc)
