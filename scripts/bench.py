"""Measure Whisper fine-tuning and decoding speed on this machine with synthetic audio."""
import sys
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_name = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-base"
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model.config.forced_decoder_ids = None

audio = [np.random.randn(16000 * 10).astype(np.float32) * 0.01 for _ in range(batch_size)]
feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
labels = processor.tokenizer(["the patient was given two tablets of paracetamol twice daily"] * batch_size,
                             return_tensors="pt").input_ids.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()
times = []
for step in range(6):
    t = time.time()
    loss = model(input_features=feats, labels=labels).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    torch.mps.synchronize() if device == "mps" else None
    times.append(time.time() - t)
train_s = np.mean(times[2:])

model.eval()
t = time.time()
with torch.no_grad():
    model.generate(input_features=feats, language="en", task="transcribe", max_new_tokens=64)
torch.mps.synchronize() if device == "mps" else None
gen_s = time.time() - t

mem = torch.mps.driver_allocated_memory() / 1e9 if device == "mps" else 0
print(f"{model_name} bs={batch_size}: train {train_s:.2f}s/step "
      f"({batch_size / train_s:.1f} clips/s), generate {gen_s:.2f}s/batch, mem {mem:.1f} GB")
