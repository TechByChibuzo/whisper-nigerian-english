"""Transcribe a manifest with a Whisper model and report WER overall, per accent and per domain.

Usage: python scripts/evaluate.py MODEL_OR_CHECKPOINT RUN_NAME [--split test] [--limit N]
"""
import argparse
import json
from pathlib import Path

import jiwer
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DATA = ROOT / "data" / "afrispeech"
torch.backends.cuda.matmul.allow_tf32 = True


def wer_table(df):
    rows = {"overall": jiwer.wer(list(df.ref_norm), list(df.hyp_norm))}
    for col in ["accent", "domain"]:
        for key, g in df.groupby(col):
            rows[f"{col}={key}"] = jiwer.wer(list(g.ref_norm), list(g.hyp_norm))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("run_name")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_repeat_ngram", type=int, default=0, help="block repeated n-grams while decoding")
    args = ap.parse_args()

    # LoRA checkpoints store their base model name in adapter_config.json
    adapter_cfg = Path(args.model) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        model = WhisperForConditionalGeneration.from_pretrained(base, dtype=torch.float32)
        model = PeftModel.from_pretrained(model, args.model).merge_and_unload()
        processor = WhisperProcessor.from_pretrained(base)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32)
        processor = WhisperProcessor.from_pretrained(args.model)
    model.to(DEVICE).eval()
    # Same normalizer for every run (fine-tuned checkpoints don't save the spelling map)
    spelling = WhisperProcessor.from_pretrained("openai/whisper-base").tokenizer.english_spelling_normalizer
    normalizer = EnglishTextNormalizer(spelling)

    df = pd.read_csv(ROOT / "data" / "afrispeech" / f"{args.split}_manifest.csv")
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)

    hyps = []
    for i in range(0, len(df), args.batch_size):
        batch = df.iloc[i:i + args.batch_size]
        audio = [sf.read(DATA / f, dtype="float32")[0] for f in batch.file]  # absolute paths pass through
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(input_features=feats.to(DEVICE), language="en", task="transcribe",
                                 max_new_tokens=200, no_repeat_ngram_size=args.no_repeat_ngram)
        hyps += processor.batch_decode(ids, skip_special_tokens=True)
        print(f"\r{len(hyps)}/{len(df)}", end="", flush=True)
    print()

    df["hypothesis"] = [h.strip() for h in hyps]
    df["ref_norm"] = df.transcript.map(normalizer)
    df["hyp_norm"] = df.hypothesis.map(normalizer)
    df = df[df.ref_norm.str.len() > 0]
    df["clip_wer"] = [jiwer.wer(r, h) for r, h in zip(df.ref_norm, df.hyp_norm)]

    out = ROOT / "results" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["ref_norm", "hyp_norm"]).to_csv(out / f"{args.split}_predictions.csv", index=False)
    table = wer_table(df)
    (out / f"{args.split}_wer.json").write_text(json.dumps(table, indent=2))
    for k, v in table.items():
        print(f"{k:20s} WER {v * 100:6.2f}%")


if __name__ == "__main__":
    main()
