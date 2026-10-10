#!/bin/bash
# Fresh construction for 6_shortage from a MIP plan, then full polish, in its own directory
# usage: fresh6.sh PLAN WORKDIR
set -u
PLAN=$1; W=$2
mkdir -p $W
I=/home/user/EID/datasets/6_shortage.json
cd /home/user/EID
python3 -u solve2.py datasets/6_shortage.json $PLAN 20 0 60 $W/f0.json > $W/f0.log 2>&1
cd /home/user/EID/work
for t in n t i c; do
  python3 -u compact.py $I $W/f0.json $t 6 200 5 $W/c_$t.json all 3 0.5 > $W/c_$t.log 2>&1 &
done
wait
python3 merge_types.py $I $W/f0.json $W/f1.json n:$W/c_n.json t:$W/c_t.json i:$W/c_i.json c:$W/c_c.json
echo '{"tlim": 120}' > $W/p.json
python3 -u recreate.py $I $W/f1.json $W/p.json $W/f2.json > $W/f2.log 2>&1
python3 -u roles2.py $I $W/f2.json $W/f3.json 600 2 > $W/f3.log 2>&1
[ -f $W/f3.json ] || cp $W/f2.json $W/f3.json
for s in 1 2; do
  nohup python3 -u lns_rr.py $I $W/f3.json $W/rr_$s.json 3000 $((s+200)) > $W/rr_$s.log 2>&1 &
done
wait
