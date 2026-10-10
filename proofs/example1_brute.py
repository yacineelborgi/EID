"""Exhaustive enumeration for 1_example: every dataset -> (unused | model j), every
mono model -> (not a source | source of target k). Scores with the official checker +
strict_ok. Prints the max legit score and one optimal solution."""
import json, itertools, sys
sys.path.insert(0, '/home/user/EID')
from test_solution import getSolutionScore
from solve import strict_ok
txt = open('/home/user/EID/datasets/1_example.json').read()
I = json.loads(txt); M = I['models']; nM = len(M); nD = len(I['datasets'])
mono = [i for i in range(nM) if len(M[i]['dataRequirements']) == 1]
best = (-1, None); cnt = 0; nvalid = 0
for dass in itertools.product(range(-1, nM), repeat=nD):
    dm = [[d, m] for d, m in enumerate(dass) if m >= 0]
    # source choices: each mono model -> -1 or a target index
    choices = []
    for s in mono:
        st = next(iter(M[s]['dataRequirements']))
        choices.append([-1] + [t for t in range(nM) if t != s and st in M[t]['dataRequirements']])
    for sc in itertools.product(*choices):
        mm = [[s, t] for s, t in zip(mono, sc) if t >= 0]
        sol = {"dataMappings": dm, "modelMappings": mm}
        cnt += 1
        if not strict_ok(I, sol):
            continue
        v, ok, _ = getSolutionScore(json.dumps(sol), txt, show_details=False)
        if ok:
            nvalid += 1
            if v > best[0]:
                best = (v, sol)
print('enumerated', cnt, 'legit valid', nvalid, 'best', best[0], json.dumps(best[1]))
