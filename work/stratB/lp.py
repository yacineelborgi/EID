import sys, time, numpy as np, highspy, exm, pickle
mg=[int(x) for x in sys.argv[1].split(',')]
mo=exm.build(marg=dict(zip('ntic',mg)), W=int(sys.argv[2]) if len(sys.argv)>2 else 800)
h=exm.highs(mo); h.setOptionValue('threads',1)
lp_int=[highspy.HighsVarType.kContinuous]*len(mo.cost)
h.changeColsIntegrality(len(mo.cost), np.arange(len(mo.cost),dtype=np.int32), np.array(lp_int))
t0=time.time(); h.run(); print('LP', h.getInfo().objective_function_value, h.modelStatusToString(h.getModelStatus()), f'{time.time()-t0:.0f}s')
sol=h.getSolution(); 
rc=np.array(sol.col_dual); x=np.array(sol.col_value); rd=np.array(sol.row_dual)
print('mass duals', {t: rd[mo.massrow[t]] for t in 'ntic'}, 'E dual', rd[[q for q,r in enumerate(mo.rows) if r[3]=='E'][0]])
pickle.dump({'rc':rc,'x':x},open('lp_%s.pkl'%sys.argv[1].replace(',','_'),'wb'))
