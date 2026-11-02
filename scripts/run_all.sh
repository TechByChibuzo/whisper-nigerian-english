#!/bin/bash
# Full experiment grid. Skips any run whose test results already exist, so it can be re-launched after a crash.
# Launch: caffeinate -i bash scripts/run_all.sh > logs/run_all.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python

evaluate() {  # evaluate MODEL_OR_CKPT RUN_NAME BATCH
  [ -f "results/$2/test_wer.json" ] && return
  echo "=== eval $2 $(date +%H:%M)"
  $PY -u scripts/evaluate.py "$1" "$2" --batch_size "$3" > "logs/eval_$2.log" 2>&1
  grep overall "logs/eval_$2.log"
}

train_eval() {  # train_eval RUN_NAME EVAL_BATCH TRAIN_ARGS...
  local run=$1 ebs=$2; shift 2
  if [ ! -f "results/$run/test_wer.json" ]; then
    echo "=== train $run $(date +%H:%M)"
    $PY -u scripts/train.py "$run" "$@" > "logs/train_$run.log" 2>&1
    tail -1 "logs/train_$run.log"
  fi
  evaluate "checkpoints/$run" "$run" "$ebs"
}

# 1. Zero-shot baselines
evaluate openai/whisper-base zeroshot_base 8
evaluate openai/whisper-small zeroshot_small 8

# 2. Data scaling with whisper-base: full fine-tuning vs LoRA at 1 h, 3 h, all (~13 h)
BASE="--model openai/whisper-base --batch_size 8 --grad_accum 2 --steps 600"
for h in 1 3; do
  train_eval base_full_${h}h 8 $BASE --hours $h
  train_eval base_lora_${h}h 8 $BASE --hours $h --lora
done
train_eval base_full_all 8 $BASE
train_eval base_lora_all 8 $BASE --lora

# 3. Bigger model: whisper-small on all data
SMALL="--model openai/whisper-small --batch_size 8 --grad_accum 2 --steps 600 --grad_ckpt"
train_eval small_full_all 8 $SMALL
train_eval small_lora_all 8 $SMALL --lora

# 4. Reference point: a much larger off-the-shelf model, zero-shot
evaluate openai/whisper-large-v3-turbo zeroshot_large_turbo 4

echo "=== ALL DONE $(date +%H:%M)"
