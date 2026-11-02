"""Download the Hausa, Igbo and Yoruba subsets of AfriSpeech-200 and unpack the audio.

Usage: python scripts/download_data.py   (writes to data/afrispeech/)
Then:  python scripts/build_manifest.py && python scripts/resample.py
"""
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent / "data" / "afrispeech"
REPO = "intronhealth/afrispeech-200"

for acc in ["hausa", "igbo", "yoruba"]:
    for split in ["train", "dev", "test"]:
        hf_hub_download(REPO, f"transcripts/{acc}/{split}.csv", repo_type="dataset", local_dir=ROOT)
        tar = hf_hub_download(REPO, f"audio/{acc}/{split}/{split}_{acc}_0.tar.gz", repo_type="dataset", local_dir=ROOT)
        with tarfile.open(tar) as t:
            t.extractall(ROOT / "wav" / acc / split, filter="data")
        print("done", acc, split, flush=True)
