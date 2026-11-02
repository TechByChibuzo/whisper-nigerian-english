"""Convert the 44.1 kHz stereo AfriSpeech clips to 16 kHz mono (Whisper's input format) and repoint the manifests."""
from multiprocessing import Pool
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent / "data" / "afrispeech"
OUT = ROOT / "wav16k"


def convert(src):
    dst = OUT / Path(src).name
    if not dst.exists():
        audio, _ = librosa.load(src, sr=16000, mono=True)
        sf.write(dst, audio, 16000)
    return str(dst)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for split in ["train", "dev", "test"]:
        path = ROOT / f"{split}_manifest.csv"
        df = pd.read_csv(path)
        with Pool(8) as pool:
            df["file"] = pool.map(convert, df.file, chunksize=32)
        df.to_csv(path, index=False)
        print(f"{split}: {len(df)} clips converted")
