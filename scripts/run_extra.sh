#!/bin/bash
# Follow-up experiments, run after run_all.sh. Skips finished runs.
# Launch: caffeinate -i bash scripts/run_extra.sh > logs/run_extra.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BASE="--model openai/whisper-base --batch_size 8 --grad_accum 2 --steps 300"

# 1. Cross-accent transfer: 3 h of a single accent (control = base_full_3h, 3 h mixed)
for acc in hausa igbo yoruba; do
  run=base_full_3h_${acc}only
  if [ ! -f "results/$run/test_wer.json" ]; then
    echo "=== train $run $(date +%H:%M)"
    $PY -u scripts/train.py $run $BASE --hours 3 --accents $acc > logs/train_$run.log 2>&1
    echo "=== eval $run $(date +%H:%M)"
    $PY -u scripts/evaluate.py checkpoints/$run $run --batch_size 8 > logs/eval_$run.log 2>&1
    grep overall logs/eval_$run.log
  fi
done

# 2. Can a decoding trick fix hallucination without training? (block repeated 3-grams)
for m in base small; do
  run=zeroshot_${m}_norepeat
  if [ ! -f "results/$run/test_wer.json" ]; then
    echo "=== eval $run $(date +%H:%M)"
    $PY -u scripts/evaluate.py openai/whisper-$m $run --batch_size 8 --no_repeat_ngram 3 > logs/eval_$run.log 2>&1
    grep overall logs/eval_$run.log
  fi
done

echo "=== EXTRA DONE $(date +%H:%M)"
