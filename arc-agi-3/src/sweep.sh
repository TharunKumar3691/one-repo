#!/bin/bash
# usage: sweep.sh label [env assignments...]; runs seeds 0..2, prints mean
label=$1; shift
tot=0
for sd in 0 1 2; do
  v=$(env "$@" timeout 1500 /tmp/claude-0/venv/bin/python /tmp/claude-0/dev/pharness.py ${MOD:-agent} 8000 60 "" $sd 2>&1 | tee /tmp/claude-0/dev/log_${label}_$sd.txt | grep TOTAL | awk '{print $2}')
  echo "  $label seed$sd $v"
  tot=$(python3 -c "print($tot+$v)")
done
python3 -c "print('$label MEAN', round($tot/3,3))"
