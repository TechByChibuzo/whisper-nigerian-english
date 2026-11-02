#!/bin/bash
#SBATCH --job-name=whisper_ng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/slurm_%j.out

# Submit from $SCRATCH/whisper_ng:   sbatch grace/job.slurm
cd $SLURM_SUBMIT_DIR
module purge
module load GCCcore/13.2.0 Python/3.11.5
source venv/bin/activate
export HF_HOME=$SCRATCH/hf_cache HF_HUB_OFFLINE=1
mkdir -p logs
nvidia-smi

# 1. Bigger model: whisper-large-v3-turbo (809M) with LoRA, same recipe as the other 13 h runs
python -u scripts/evaluate.py openai/whisper-large-v3-turbo zeroshot_large_turbo --batch_size 16
python -u scripts/train.py large_lora_all --model openai/whisper-large-v3-turbo --lora \
    --batch_size 8 --grad_accum 2 --steps 600 --grad_ckpt
python -u scripts/evaluate.py checkpoints/large_lora_all large_lora_all --batch_size 16

# 2. Does longer training help? whisper-small, full fine-tuning, 1500 steps instead of 600
python -u scripts/train.py small_full_all_1500 --model openai/whisper-small \
    --batch_size 16 --grad_accum 1 --steps 1500
python -u scripts/evaluate.py checkpoints/small_full_all_1500 small_full_all_1500 --batch_size 16

echo "JOB DONE"
