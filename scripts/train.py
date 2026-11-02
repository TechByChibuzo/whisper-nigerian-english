"""Fine-tune Whisper on (a subset of) the AfriSpeech Nigerian-accent manifest.

Every run trains for the same number of optimizer steps, tracks loss on a fixed dev subset,
and keeps the checkpoint with the lowest dev loss (guards against overfitting on small subsets).

Usage: python scripts/train.py RUN_NAME [--model openai/whisper-base] [--hours 3] [--lora]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "afrispeech"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True


def subset_by_hours(df, hours, seed):
    """Random clips per accent until each accent reaches its proportional share of `hours`."""
    if hours is None:
        return df
    share = df.groupby("accent").duration.sum() / df.duration.sum()
    parts = []
    for acc, g in df.groupby("accent"):
        g = g.sample(frac=1, random_state=seed)
        parts.append(g[g.duration.cumsum() <= hours * 3600 * share[acc]])
    return pd.concat(parts)


def make_batch(rows, processor, start_id):
    audio = [sf.read(DATA / f, dtype="float32")[0] for f in rows.file]  # absolute paths pass through
    feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    tok = processor.tokenizer(list(rows.transcript), padding=True, return_tensors="pt")
    labels = tok.input_ids.masked_fill(tok.attention_mask == 0, -100)
    if (labels[:, 0] == start_id).all():  # the model prepends <|startoftranscript|> itself
        labels = labels[:, 1:]
    return feats.to(DEVICE), labels.to(DEVICE)


@torch.no_grad()
def dev_loss(model, dev, processor, start_id, bs):
    model.eval()
    losses = []
    for i in range(0, len(dev), bs):
        feats, labels = make_batch(dev.iloc[i:i + bs], processor, start_id)
        losses.append(model(input_features=feats, labels=labels).loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--hours", type=float, default=None, help="training subset size; default = all")
    ap.add_argument("--accents", default=None, help="comma-separated accents to train on; default = all")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_ckpt", action="store_true", help="recompute activations to save memory")
    args = ap.parse_args()
    args.lr = args.lr or (1e-3 if args.lora else 1e-5)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = ROOT / "checkpoints" / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    start_id = model.config.decoder_start_token_id
    if args.grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if args.lora:
            model.enable_input_require_grads()  # LoRA + checkpointing needs grads flowing from the inputs

    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    model.to(DEVICE).train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    train = pd.read_csv(DATA / "train_manifest.csv")
    if args.accents:
        train = train[train.accent.isin(args.accents.split(","))]
    train = subset_by_hours(train, args.hours, args.seed)
    dev = pd.read_csv(DATA / "dev_manifest.csv").sample(n=240, random_state=0)
    print(f"{args.run_name}: {len(train)} clips / {train.duration.sum() / 3600:.2f} h, "
          f"trainable {n_train / 1e6:.1f}M of {n_total / 1e6:.1f}M params, lr {args.lr}, device {DEVICE}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * args.steps), args.steps)

    def batches():
        epoch = 0
        while True:
            order = train.sample(frac=1, random_state=args.seed + epoch)
            for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
                yield order.iloc[i:i + args.batch_size]
            epoch += 1

    log = {"args": vars(args), "train_hours": train.duration.sum() / 3600, "train_clips": len(train),
           "trainable_params": n_train, "total_params": n_total, "history": []}
    best = dev_loss(model, dev, processor, start_id, args.batch_size)
    log["history"].append({"step": 0, "dev_loss": best})
    print(f"step 0 dev_loss {best:.4f}")

    gen, t0, running = batches(), time.time(), []
    for step in range(1, args.steps + 1):
        for _ in range(args.grad_accum):
            feats, labels = make_batch(next(gen), processor, start_id)
            loss = model(input_features=feats, labels=labels).loss / args.grad_accum
            loss.backward()
            running.append(loss.item() * args.grad_accum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()

        if step % args.eval_every == 0 or step == args.steps:
            dl = dev_loss(model, dev, processor, start_id, args.batch_size)
            tl = float(np.mean(running))
            running = []
            log["history"].append({"step": step, "train_loss": tl, "dev_loss": dl})
            mark = ""
            if dl < best:
                best, mark = dl, " *saved"
                model.save_pretrained(out)
                processor.save_pretrained(out)
            print(f"step {step} train_loss {tl:.4f} dev_loss {dl:.4f} "
                  f"({(time.time() - t0) / 60:.1f} min){mark}", flush=True)

    if not (out / "preprocessor_config.json").exists():  # dev loss never improved: keep the base model
        model.save_pretrained(out)
        processor.save_pretrained(out)
    log["best_dev_loss"], log["minutes"] = best, (time.time() - t0) / 60
    (out / "train_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
