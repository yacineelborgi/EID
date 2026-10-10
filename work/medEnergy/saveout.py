"""usage: saveout.py instance_name sol.json cur_score -> verify (test_solution, strict_ok, int energy) and write to out_lb if better"""
import sys, json, os, subprocess
os.chdir('/home/user/EID'); sys.path.insert(0, '/home/user/EID')
from solve import strict_ok
from test_solution import getSolutionScore
name, f, cur = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = f'/home/user/EID/datasets/{name}.json'; txt = open(ds).read(); inst = json.loads(txt)
def sc_of(path):
    if not os.path.exists(path): return -1
    S = json.load(open(path))
    out = subprocess.run(['python3', '/home/user/EID/test_solution.py', '-d', ds, '-s', path], capture_output=True, text=True).stdout
    if 'valid' not in out or 'invalid' in out.lower(): return -1
    sc, ok, _ = getSolutionScore(json.dumps(S), txt, False)
    if not (ok and strict_ok(inst, S)): return -1
    M = inst['models']; used = set(m for _, m in S['dataMappings']) | set(a for a, b in S['modelMappings']) | set(b for a, b in S['modelMappings'])
    if sum(M[i]['energyCost'] for i in used) > inst['energyCap']: return -1
    return sc
new = sc_of(f); outp = f'/tmp/claude-0/-home-user-EID/33ea8f0e-6018-51eb-9751-49d6359bff2a/scratchpad/out_lb/{name}_submission.json'; old = sc_of(outp)
print('new', new, 'out_lb', old, 'cur', cur)
if new > max(old, cur):
    tmp = outp + f'.tmp{os.getpid()}'; json.dump(json.load(open(f)), open(tmp, 'w')); os.replace(tmp, outp); print('WROTE', outp, new)
