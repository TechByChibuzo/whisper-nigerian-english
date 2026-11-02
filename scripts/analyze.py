"""Turn results/ into slide material: summary table, bootstrap CIs, hallucination stats, figures and examples.

Usage: python scripts/analyze.py   (writes to analysis/)
"""
import json
import re
from pathlib import Path

import jiwer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", ROOT / "analysis"
OUT.mkdir(exist_ok=True)

# Reference palette (dataviz skill), light mode
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ACCENT_COLOR = {"hausa": BLUE, "igbo": ORANGE, "yoruba": AQUA}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 13, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})

normalizer = EnglishTextNormalizer(
    WhisperProcessor.from_pretrained("openai/whisper-base").tokenizer.english_spelling_normalizer)

LABELS = {
    "zeroshot_base": "Whisper-base, zero-shot",
    "zeroshot_base_norepeat": "Whisper-base, zero-shot + repeat-block",
    "zeroshot_small": "Whisper-small, zero-shot",
    "zeroshot_small_norepeat": "Whisper-small, zero-shot + repeat-block",
    "zeroshot_large_turbo": "Whisper-large-v3-turbo, zero-shot",
    "base_full_1h": "Whisper-base, full FT, 1 h", "base_lora_1h": "Whisper-base, LoRA, 1 h",
    "base_full_3h": "Whisper-base, full FT, 3 h", "base_lora_3h": "Whisper-base, LoRA, 3 h",
    "base_full_all": "Whisper-base, full FT, 13 h", "base_lora_all": "Whisper-base, LoRA, 13 h",
    "small_full_all": "Whisper-small, full FT, 13 h", "small_lora_all": "Whisper-small, LoRA, 13 h",
    "small_full_all_1500": "Whisper-small, full FT, 13 h, 1500 steps",
    "large_lora_all_lr1e4": "Whisper-large-v3-turbo, LoRA, 13 h",
    "large_lora_all": "Whisper-large-v3-turbo, LoRA, 13 h, lr 1e-3 (diverged)",
    "base_full_3h_hausaonly": "Whisper-base, full FT, 3 h Hausa only",
    "base_full_3h_igboonly": "Whisper-base, full FT, 3 h Igbo only",
    "base_full_3h_yorubaonly": "Whisper-base, full FT, 3 h Yoruba only",
}


# Dictated punctuation: speakers say "slash", "full stop"... but references write symbols (or nothing)
SPOKEN_PUNCT = re.compile(r"\b(slash|full ?stop|comma|colon|semi ?colon|semicolom|column|period|new line|"
                          r"next line|dash|hyphen|bracket|open bracket|close bracket)\b")


def load(run):
    df = pd.read_csv(RES / run / "test_predictions.csv")
    df["hypothesis"] = df.hypothesis.fillna("")
    ref = df.transcript.map(normalizer)
    hyp = df.hypothesis.map(normalizer)
    out = [jiwer.process_words(r, h) for r, h in zip(ref, hyp)]
    df["errors"] = [o.substitutions + o.deletions + o.insertions for o in out]
    df["ref_words"] = [len(r.split()) for r in ref]
    df["hyp_words"] = [len(h.split()) for h in hyp]
    df["spoken_punct"] = hyp.str.contains(SPOKEN_PUNCT)
    # Punctuation-neutral scoring: drop spoken-punctuation words from both sides
    ref_np = [" ".join(SPOKEN_PUNCT.sub(" ", r).split()) or "x" for r in ref]
    hyp_np = [" ".join(SPOKEN_PUNCT.sub(" ", h).split()) for h in hyp]
    df["errors_np"] = [jiwer.process_words(r, h).wer * len(r.split()) for r, h in zip(ref_np, hyp_np)]
    df["ref_words_np"] = [len(r.split()) for r in ref_np]
    # Runaway output: hypothesis far longer than the reference (repetition loops, invented text)
    df["hallucination"] = df.hyp_words > 2 * df.ref_words + 5
    return df


def wer(df):
    return df.errors.sum() / df.ref_words.sum()


def speaker_bootstrap(df, n=1000, seed=0):
    """95% CI for corpus WER, resampling whole speakers (clips from one speaker are correlated)."""
    rng = np.random.default_rng(seed)
    g = df.groupby("user_ids")[["errors", "ref_words"]].sum().to_numpy()
    idx = rng.integers(0, len(g), size=(n, len(g)))
    s = g[idx].sum(axis=1)
    w = s[:, 0] / s[:, 1]
    return np.percentile(w, 2.5), np.percentile(w, 97.5)


runs = [r for r in LABELS if (RES / r / "test_predictions.csv").exists()]
data = {r: load(r) for r in runs}

# ---------- summary table ----------
rows = []
for r in runs:
    df = data[r]
    lo, hi = speaker_bootstrap(df)
    log_path = ROOT / "checkpoints" / r / "train_log.json"
    tp = json.loads(log_path.read_text())["trainable_params"] if log_path.exists() else 0
    rows.append({
        "run": r, "label": LABELS[r], "WER": wer(df), "CI_low": lo, "CI_high": hi,
        "median_clip_WER": df.errors.div(df.ref_words.clip(lower=1)).median(),
        "WER_no_halluc": wer(df[~df.hallucination]), "halluc_rate": df.hallucination.mean(),
        "WER_punct_neutral": df.errors_np.sum() / df.ref_words_np.sum(),
        "WER_pn_no_halluc": df[~df.hallucination].errors_np.sum() / df[~df.hallucination].ref_words_np.sum(),
        "spoken_punct_rate": df.spoken_punct.mean(),
        **{f"WER_{a}": wer(g) for a, g in df.groupby("accent")},
        **{f"WER_{d}": wer(g) for d, g in df.groupby("domain")},
        "trainable_M": tp / 1e6,
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "summary.csv", index=False)
pct = summary.copy()
for c in pct.columns:
    if c.startswith(("WER", "CI", "median", "halluc", "spoken")):
        pct[c] = (pct[c] * 100).round(1)
print(pct.drop(columns=["run"]).to_string(index=False))
S = summary.set_index("run")


def ref_line(ax, run, text, color=MUTED, style="--", below=False):
    # Label sits on the line between the 3 h and 13 h points, clear of the data labels
    if run in S.index:
        y = S.loc[run, "WER"] * 100
        ax.axhline(y, color=color, ls=style, lw=1.5)
        ax.text(4.2, y - 0.6 if below else y + 0.6, f"{text} ({y:.1f}%)", va="top" if below else "bottom",
                ha="left", color=INK2, fontsize=11)


# ---------- fig 1: data scaling ----------
fig, ax = plt.subplots(figsize=(9, 5.2))
hours = {"1h": None, "3h": None, "all": None}
for kind, color, name in [("full", BLUE, "Full fine-tuning"), ("lora", ORANGE, "LoRA (3% of params)")]:
    pts = [(json.loads((ROOT / "checkpoints" / f"base_{kind}_{h}" / "train_log.json").read_text())["train_hours"],
            S.loc[f"base_{kind}_{h}"]) for h in hours if f"base_{kind}_{h}" in S.index]
    if not pts:
        continue
    x = [p[0] for p in pts]
    y = [p[1].WER * 100 for p in pts]
    err = [[(p[1].WER - p[1].CI_low) * 100 for p in pts], [(p[1].CI_high - p[1].WER) * 100 for p in pts]]
    ax.errorbar(x, y, yerr=err, color=color, lw=2, marker="o", ms=8, capsize=4, label=name)
    ax.annotate(f"{y[-1]:.1f}%", (x[-1], y[-1]), textcoords="offset points", xytext=(10, 0), va="center",
                color=INK, fontsize=12)
ax.set_xscale("log")
ax.set_xticks([1, 3, 13.2])
ax.set_xticklabels(["1 h", "3 h", "13 h"])
ax.set_xlim(0.8, 17)
ax.set_xlabel("Hours of Nigerian-accented training speech (log scale)")
ax.set_ylabel("Test word error rate (%)")
ax.set_ylim(20, 62)
ref_line(ax, "zeroshot_base", "base, zero-shot")
ref_line(ax, "zeroshot_small", "small, zero-shot", style=":")
ref_line(ax, "zeroshot_large_turbo", "large-v3-turbo, zero-shot", style="-.", below=True)
ax.set_title("Whisper-base: errors fall steadily with more Nigerian speech", loc="left", fontsize=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93))
fig.tight_layout()
fig.savefig(OUT / "fig_scaling.png", bbox_inches="tight")
plt.close(fig)

# ---------- fig 2: per-accent before/after ----------
compare = [r for r in ["zeroshot_base", "base_full_all", "zeroshot_small", "small_full_all", "zeroshot_large_turbo", "large_lora_all_lr1e4"] if r in S.index]
accents = ["hausa", "igbo", "yoruba"]
fig, ax = plt.subplots(figsize=(11, 5.6))
w = 0.8 / len(compare)
shades = {"zeroshot_base": "#b9d3f2", "base_full_all": BLUE, "zeroshot_small": "#f7c3ad", "small_full_all": ORANGE, "zeroshot_large_turbo": "#a9e4cf", "large_lora_all_lr1e4": AQUA}
for i, r in enumerate(compare):
    vals = [S.loc[r, f"WER_{a}"] * 100 for a in accents]
    xs = np.arange(3) + (i - (len(compare) - 1) / 2) * w
    ax.bar(xs, vals, width=w - 0.02, color=shades[r], label=LABELS[r])
    for xx, v in zip(xs, vals):
        ax.text(xx, v + 0.8, f"{v:.0f}", ha="center", fontsize=10, color=INK2)
ax.set_xticks(range(3))
ax.set_xticklabels([a.capitalize() + "-accented English" for a in accents])
ax.set_ylabel("Test word error rate (%)")
ax.set_title("Fine-tuning helps every accent; Hausa gains the most", loc="left", fontsize=14)
ax.legend(fontsize=10, loc="upper right", ncol=3)
ax.set_ylim(0, 92)
fig.tight_layout()
fig.savefig(OUT / "fig_accents.png", bbox_inches="tight")
plt.close(fig)

# ---------- fig 2b: zero-shot vs fine-tuned, by model size ----------
sizes = [("Whisper-base\n74M", "zeroshot_base", "base_full_all", "full FT"),
         ("Whisper-small\n244M", "zeroshot_small", "small_full_all", "full FT"),
         ("Whisper-large-v3-turbo\n809M", "zeroshot_large_turbo", "large_lora_all_lr1e4", "LoRA")]
sizes = [s for s in sizes if s[1] in S.index and s[2] in S.index]
if sizes:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(sizes))
    for off, col, color, name in [(-0.2, 1, MUTED, "Off the shelf (zero-shot)"),
                                  (0.2, 2, BLUE, "Fine-tuned on 13 h Nigerian speech")]:
        vals = [S.loc[s[col], "WER"] * 100 for s in sizes]
        lo = [(S.loc[s[col], "WER"] - S.loc[s[col], "CI_low"]) * 100 for s in sizes]
        hi = [(S.loc[s[col], "CI_high"] - S.loc[s[col], "WER"]) * 100 for s in sizes]
        ax.bar(x + off, vals, width=0.38, color=color, label=name, yerr=[lo, hi], capsize=4,
               error_kw={"ecolor": INK2, "lw": 1.2})
        for xx, v, h in zip(x + off, vals, hi):
            ax.text(xx, v + h + 1, f"{v:.1f}%", ha="center", fontsize=12, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] + f" ({s[3]})" for s in sizes])
    ax.set_ylabel("Test word error rate (%)")
    ax.set_ylim(0, 70)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_title("Fine-tuning cuts errors 44–51% at every model size", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig_models.png", bbox_inches="tight")
    plt.close(fig)

# ---------- fig 3: hallucination rate ----------
hrun = [r for r in ["zeroshot_base", "zeroshot_base_norepeat", "zeroshot_small", "zeroshot_large_turbo",
                    "base_full_1h", "base_full_all", "small_full_all", "large_lora_all_lr1e4"] if r in S.index]
fig, ax = plt.subplots(figsize=(9, 4.8))
vals = [S.loc[r, "halluc_rate"] * 100 for r in hrun]
colors = [MUTED if r.startswith("zeroshot") else BLUE for r in hrun]
ax.barh(range(len(hrun)), vals, color=colors, height=0.6)
for i, v in enumerate(vals):
    ax.text(v + 0.02, i, f"{v:.1f}%", va="center", fontsize=11, color=INK)
ax.set_yticks(range(len(hrun)))
ax.set_yticklabels([LABELS[r] for r in hrun], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("Test clips with runaway output (% of 1,185)")
ax.set_title("Runaway loops are rare; fine-tuning or a repeat-block both cut them", loc="left", fontsize=14)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(OUT / "fig_hallucination.png", bbox_inches="tight")
plt.close(fig)

# ---------- fig 3b: where does the zero-shot error come from? ----------
ft_run = "small_full_all" if "small_full_all" in S.index else "base_full_all"
for zs_run, ft_run_d in [("zeroshot_base", "base_full_all"), ("zeroshot_small", ft_run)]:
    if zs_run not in S.index or ft_run_d not in S.index or ft_run_d.split("_")[0] != zs_run.split("_")[1]:
        continue
    z, f = S.loc[zs_run], S.loc[ft_run_d]
    steps = [("Zero-shot (as reported)", z.WER), ("...minus hallucination loops", z.WER_no_halluc),
             ("...minus dictated-punctuation words", z.WER_pn_no_halluc),
             ("Fine-tuned 13 h, same scoring", f.WER_pn_no_halluc)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    vals = [v * 100 for _, v in steps]
    ax.barh(range(4), vals, color=[MUTED, MUTED, MUTED, BLUE], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=12, color=INK)
    ax.set_yticks(range(4))
    ax.set_yticklabels([s for s, _ in steps])
    ax.invert_yaxis()
    ax.set_xlabel("Test word error rate (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.15)
    size = zs_run.split("_")[1]
    ax.set_title(f"Whisper-{size}: most of the improvement is real, not a scoring artifact", loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_decomposition_{size}.png", bbox_inches="tight")
    plt.close(fig)

# ---------- fig 4: loss curves (overfitting vs data size) ----------
fig, ax = plt.subplots(figsize=(9, 4.8))
for h, color, name in [("1h", ORANGE, "1 h"), ("all", BLUE, "13 h")]:
    p = ROOT / "checkpoints" / f"base_full_{h}" / "train_log.json"
    if not p.exists():
        continue
    hist = json.loads(p.read_text())["history"]
    tr = [(e["step"], e["train_loss"]) for e in hist if "train_loss" in e]
    dv = [(e["step"], e["dev_loss"]) for e in hist]
    ax.plot(*zip(*tr), color=color, lw=2, ls="--", label=f"{name}: train loss")
    ax.plot(*zip(*dv), color=color, lw=2, marker="o", ms=6, label=f"{name}: validation loss")
ax.set_xlabel("Optimizer step (batch of 16 clips)")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("With 1 h the model memorizes after ~100 steps; with 13 h it keeps generalizing",
             loc="left", fontsize=13)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig_loss_curves.png", bbox_inches="tight")
plt.close(fig)

# ---------- fig 5: cross-accent transfer matrix ----------
xruns = [("base_full_3h_hausaonly", "Hausa only"), ("base_full_3h_igboonly", "Igbo only"),
         ("base_full_3h_yorubaonly", "Yoruba only"), ("base_full_3h", "Mixed (control)")]
xruns = [x for x in xruns if x[0] in S.index]
if len(xruns) > 2:
    M = np.array([[S.loc[r, f"WER_{a}"] * 100 for a in accents] for r, _ in xruns])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(M, cmap="Blues", vmin=M.min() - 5, vmax=M.max())
    for i in range(M.shape[0]):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=12,
                    color="white" if M[i, j] > (M.min() + M.max()) / 2 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([a.capitalize() for a in accents])
    ax.set_yticks(range(len(xruns)))
    ax.set_yticklabels([n for _, n in xruns])
    ax.set_xlabel("Test speakers' accent")
    ax.set_ylabel("3 h of training data from")
    ax.grid(False)
    ax.set_title("One accent’s data helps all three almost equally\n(test WER %, Whisper-base, 3 h; zero-shot: 74 / 51 / 58)",
                 loc="left", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_accent.png", bbox_inches="tight")
    plt.close(fig)

# ---------- examples ----------
best = "small_full_all" if "small_full_all" in data else "base_full_all"
zs, ft = data["zeroshot_base"], data[best]
m = zs[["file", "transcript", "accent", "domain", "errors", "ref_words", "hypothesis", "hallucination"]].merge(
    ft[["file", "hypothesis", "errors"]], on="file", suffixes=("_zs", "_ft"))
m["gain"] = (m.errors_zs - m.errors_ft) / m.ref_words.clip(lower=1)
lines = [f"# Examples: zero-shot Whisper-base vs. {LABELS[best]}\n"]


def show(title, df):
    lines.append(f"\n## {title}\n")
    for _, r in df.iterrows():
        lines.append(f"- [{r.accent}, {r.domain}]\n  - REF: {r.transcript}\n  - ZERO-SHOT: {r.hypothesis_zs[:250]}"
                     f"\n  - FINE-TUNED: {r.hypothesis_ft[:250]}\n")


short = m[(m.ref_words.between(6, 25)) & (~m.hallucination)]
show("Big fixes", short.sort_values("gain", ascending=False).head(10))
show("Hallucinations cured", m[m.hallucination & (m.errors_ft <= m.ref_words * 0.3)].head(8))
show("Still wrong after fine-tuning", m[m.ref_words.between(5, 25)].assign(
    r=lambda d: d.errors_ft / d.ref_words).sort_values("r", ascending=False).head(10))
show("Fine-tuning made it worse", m[m.ref_words >= 5].sort_values("gain").head(5))
(OUT / "examples.md").write_text("\n".join(lines))
print(f"\nWrote figures, summary.csv and examples.md to {OUT}")
