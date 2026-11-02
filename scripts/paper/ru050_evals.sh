#!/bin/bash
# RU050 evaluations for the paper (scripts/paper/side_view.py --worlds 0: 4096 games, deterministic, reset seed 0;
# how every game ended + game length). Shard K of N (by line); existing outputs are skipped.
#   outcomes/RUN_IT.json   self-play at every 1000 iterations, 0..40000, both seeds
#   cross/P_vs_E_IT.json   pursuer-seat checkpoint P vs evader-seat checkpoint E (rules.learner_role=pursuer),
#                          s0/s1 in all four pairings at 20000 and 40000
#
#   bash scripts/paper/ru050_evals.sh K N
K=$1; N=$2
C=logs/mofo/tag/tag_chase_adapt; O=logs/plots/ru050; mkdir -p $O/outcomes $O/cross
{
  for s in 0 1; do for it in $(seq 0 1000 40000); do echo "self $s $it"; done; done
  for it in 20000 40000; do for p in 0 1; do for e in 0 1; do echo "cross $p $e $it"; done; done; done
} | awk -v k=$K -v n=$N 'NR % n == k' | while read kind a b c; do
  if [ $kind = self ]; then
    out=$O/outcomes/RU050_s${a}_$b
    [ -f $out.json ] || .venv/bin/python scripts/paper/side_view.py $C/RU050_s$a $C/RU050_s$a/model_$b.pt $out \
      --worlds 0 --envs 4096 > /dev/null
  else
    out=$O/cross/s${a}_vs_s${b}_$c
    [ -f $out.json ] || .venv/bin/python scripts/paper/side_view.py $C/RU050_s0 $C/RU050_s$a/model_$c.pt $out \
      --opp $C/RU050_s$b/model_$c.pt --set rules.learner_role=pursuer --worlds 0 --envs 4096 > /dev/null
  fi
  [ -f $out.json ] && echo "done $out" || echo "FAILED $out"
done
