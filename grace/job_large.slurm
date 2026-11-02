#!/bin/bash
#SBATCH --job-name=whisper_ng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

# Rerun of the large-turbo LoRA at a 10x lower learning rate (lr 1e-3 diverged and collapsed)
python -u scripts/train.py large_lora_all_lr1e4 --model openai/whisper-large-v3-turbo --lora \
    --batch_size 8 --grad_accum 2 --steps 600 --grad_ckpt --lr 1e-4
python -u scripts/evaluate.py checkpoints/large_lora_all_lr1e4 large_lora_all_lr1e4 --batch_size 16

echo "JOB DONE"
