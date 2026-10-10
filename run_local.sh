#!/bin/bash
# Lance la recherche sur le fichier 6 sur ton PC (Gurobi si la licence est installée, sinon HiGHS).
# usage : ./run_local.sh [nb_processus] [threads_par_processus]
#   ex. 16 coeurs : ./run_local.sh 4 4   (4 recherches en parallele, 4 threads Gurobi chacune)
set -u
NPROC=${1:-4}
THR=${2:-4}
cd "$(dirname "$0")"
ROOT=$(pwd)
OUT=$ROOT/local
mkdir -p "$OUT"
if python3 -c "import gurobipy as g; m=g.Model(); m.addVars(3000); m.update(); m.optimize()" >/dev/null 2>&1; then
  export SOLVER=gurobi; echo "Solveur : Gurobi"
else
  export SOLVER=highs; echo "Solveur : HiGHS (licence Gurobi absente ou invalide)"
fi
export THREADS=$THR
export BIG_TLIM=${BIG_TLIM:-600}
export REFILL_MIN=${REFILL_MIN:-200}
export REFILL_MAX=${REFILL_MAX:-500}
[ -f "$OUT/start6.json" ] || cp solutions/6_shortage_submission.json "$OUT/start6.json"
cd work
for s in $(seq 1 "$NPROC"); do
  mode=big; [ $((s % 2)) -eq 0 ] && mode=fast
  nohup python3 -u lns_rr.py "$ROOT/datasets/6_shortage.json" "$OUT/start6.json" "$OUT/rr_$s.json" 1000000 $((1000 + s)) $mode \
    > "$OUT/rr_$s.log" 2>&1 &
  echo "recherche $s lancee (mode $mode) -> $OUT/rr_$s.log"
done
echo
echo "Meilleur score : cat $OUT/rr_shared.json.score   (fichier : $OUT/rr_shared.json)"
echo "Arreter tout   : pkill -f lns_rr.py"
