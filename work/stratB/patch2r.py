s=open('lnsR.py').read()
s=s.replace("""        fb = set(rng.sample(mo.sbins, NB))""","""        fb = set(rng.sample(mo.sbins, NB)) if FT_ is None else {b for b in mo.sbins if b[1] in FT_}""")
s=s.replace("""XFIX = None""","""XFIX = None; FT_ = None
E1 = [i for i in range(len(M)) if M[i]['energyCost'] <= int(os.environ.get('EMAX', '1000'))]
MODE = os.environ.get('MODE', 'rand')""")
s=s.replace("""    free = set(rng.sample(tr, min(int(NT*SZ), len(tr)))) | set(rng.sample(un, min(int(NU*SZ), len(un))))""","""    free = set(rng.sample(tr, min(int(NT*SZ), len(tr)))) | set(rng.sample(un, min(int(NU*SZ), len(un))))
    if MODE == 'e1':
        FT_ = rng.choice(['i', 'n', 't', 'c', 'in', 'nc', 'tc', 'it'])
        free = set(E1) | set(rng.sample(tr, min(int(NT*SZ), len(tr))))""")
s=s.replace("""SZ {SZ:.2f}')""","""SZ {SZ:.2f} FT {FT_}')""")
open('lnsR.py','w').write(s)
