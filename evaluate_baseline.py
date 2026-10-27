"""
Scores the RAW NOISY audio (no denoising at all) against clean references,
using the same eval set as evaluate.py. This is the baseline your trained
model needs to beat -- run this once and keep the numbers for your PPT's
"baseline vs. ours" comparison table.
"""

import glob
import os
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

SR = 16000


def score_pair(clean_np, enhanced_np, sr=SR):
    min_len = min(len(clean_np), len(enhanced_np))
    clean_np = clean_np[:min_len]
    enhanced_np = enhanced_np[:min_len]
    pesq_score = pesq(sr, clean_np, enhanced_np, 'wb')
    stoi_score = stoi(clean_np, enhanced_np, sr, extended=False)
    return pesq_score, stoi_score


def find_eval_pairs(eval_dir="data/eval"):
    noisy_files = sorted(glob.glob(f"{eval_dir}/noisy_*.wav"))
    pairs = []
    for noisy_path in noisy_files:
        idx = noisy_path.split("noisy_")[-1]
        clean_path = f"{eval_dir}/clean_{idx}"
        if os.path.exists(clean_path):
            pairs.append((noisy_path, clean_path))
    return pairs


if __name__ == "__main__":
    pairs = find_eval_pairs()
    print(f"Found {len(pairs)} eval pairs. Scoring RAW NOISY (no denoising) vs clean...\n")

    results = []
    for noisy_path, clean_path in pairs:
        noisy_np, _ = sf.read(noisy_path)
        clean_np, _ = sf.read(clean_path)
        if noisy_np.ndim > 1:
            noisy_np = noisy_np.mean(axis=1)
        if clean_np.ndim > 1:
            clean_np = clean_np.mean(axis=1)

        p, s = score_pair(clean_np, noisy_np)
        results.append((noisy_path, p, s))
        print(f"{noisy_path}: PESQ={p:.3f}  STOI={s:.3f}")

    scores = np.array([(p, s) for _, p, s in results])
    print(f"\nBASELINE (no denoising) Mean PESQ={scores[:, 0].mean():.3f}  Mean STOI={scores[:, 1].mean():.3f}")
