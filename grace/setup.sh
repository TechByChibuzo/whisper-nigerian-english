#!/bin/bash
# One-time setup on the Grace LOGIN node (it has internet; compute nodes may not).
# Run from $SCRATCH/whisper_ng:   bash grace/setup.sh
set -e
cd "$(dirname "$0")/.."

module purge
module load GCCcore/13.2.0 Python/3.11.5

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio transformers accelerate peft jiwer soundfile librosa pandas matplotlib

# Cache the models on scratch so the GPU job never needs the internet
export HF_HOME=$SCRATCH/hf_cache
python - <<'EOF'
from transformers import WhisperForConditionalGeneration, WhisperProcessor
for m in ["openai/whisper-small", "openai/whisper-base", "openai/whisper-large-v3-turbo"]:
    WhisperProcessor.from_pretrained(m)
    WhisperForConditionalGeneration.from_pretrained(m)
    print("cached", m)
EOF

python -c "import torch; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)"
echo "SETUP DONE"
