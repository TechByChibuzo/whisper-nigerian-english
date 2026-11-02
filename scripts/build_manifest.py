"""Match extracted AfriSpeech wav files to transcript rows and write one manifest CSV per split."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "afrispeech"
ACCENTS = ["yoruba", "igbo", "hausa"]

wavs = {p.name: str(p) for p in (ROOT / "wav").rglob("*.wav")}
print(f"{len(wavs)} wav files on disk")

for split in ["train", "dev", "test"]:
    rows = []
    for acc in ACCENTS:
        df = pd.read_csv(ROOT / "transcripts" / acc / f"{split}.csv")
        df["file"] = df.audio_paths.map(lambda p: wavs.get(Path(p).name))
        rows.append(df[df.file.notna()])
    out = pd.concat(rows, ignore_index=True)
    out["transcript"] = out.transcript.str.strip()
    out = out[out.duration <= 30]  # Whisper's input window is 30 s
    out[["file", "transcript", "accent", "domain", "duration", "user_ids", "gender"]].to_csv(
        ROOT / f"{split}_manifest.csv", index=False)
    summary = out.groupby("accent").duration.agg(["count", "sum"])
    summary["sum"] = (summary["sum"] / 3600).round(2)
    print(f"\n{split}: {len(out)} clips, {out.duration.sum() / 3600:.2f} h\n{summary.rename(columns={'sum': 'hours'})}")
