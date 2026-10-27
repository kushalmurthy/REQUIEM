"""
Builds a real held-out evaluation set: mixes clean speech (a slice reserved
for eval, not used in training) with noise from the MAD *test* CSV (a
genuinely separate split from datalabeltrain.csv, so this isn't data the
model has seen). Writes noisy/clean wav pairs to data/eval/, which
evaluate.py then scores automatically.

IMPORTANT: run this once now to generate your eval set. Going forward,
exclude these same clean files from train.py's clean_files list (see the
note printed at the end) so you're not accidentally training on your own
test set.
"""

import os
import random
import soundfile as sf
import numpy as np
import glob

from mad_loader import load_mad_files

SR = 16000
N_EVAL_PAIRS = 15
SNR_DB = 5.0  # fixed, moderate difficulty -- realistic single-number benchmark
EVAL_DIR = "data/eval"


def load_segment(path, seg_len, sr=SR):
    audio_np, file_sr = sf.read(path)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    if file_sr != sr:
        # simple resample via soundfile isn't available; skip files with wrong sr
        return None
    if len(audio_np) < seg_len:
        audio_np = np.pad(audio_np, (0, seg_len - len(audio_np)))
    else:
        start = random.randint(0, len(audio_np) - seg_len)
        audio_np = audio_np[start:start + seg_len]
    return audio_np.astype(np.float32)


def mix_at_snr(clean, noise, snr_db):
    noise = noise[:len(clean)] if len(noise) >= len(clean) else np.pad(noise, (0, len(clean) - len(noise)))
    clean_power = np.mean(clean ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
    return clean + scale * noise


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    # held-out clean speech: LAST 20 files, reserved for eval only
    all_clean = sorted(glob.glob("data/LibriSpeech/dev-clean/**/*.flac", recursive=True))
    eval_clean_files = all_clean[-20:]
    print(f"Reserved {len(eval_clean_files)} clean files for eval "
          f"(the last 20 from your dev-clean list).")

    # held-out noise: MAD TEST split (separate from what you trained on)
    test_noise_files, test_noise_labels = load_mad_files(
        "datalabeltes.csv", "data/MAD/MAD_dataset"
    )
    print(f"Loaded {len(test_noise_files)} noise files from MAD test split.")

    seg_len = int(4.0 * SR)  # 4-second clips, matching training segment length

    made = 0
    for i, clean_path in enumerate(eval_clean_files[:N_EVAL_PAIRS]):
        clean = load_segment(clean_path, seg_len)
        if clean is None:
            continue

        noise_path = random.choice(test_noise_files)
        noise_raw, noise_sr = sf.read(noise_path)
        if noise_raw.ndim > 1:
            noise_raw = noise_raw.mean(axis=1)
        if noise_sr != SR:
            continue  # skip mismatched sample rates for simplicity here
        noise = load_segment(noise_path, seg_len)
        if noise is None:
            continue

        noisy = mix_at_snr(clean, noise, SNR_DB)

        clean_out = os.path.join(EVAL_DIR, f"clean_{i:02d}.wav")
        noisy_out = os.path.join(EVAL_DIR, f"noisy_{i:02d}.wav")
        sf.write(clean_out, clean, SR)
        sf.write(noisy_out, noisy, SR)
        made += 1

    print(f"\nWrote {made} eval pairs to {EVAL_DIR}/")
    print(f"\nIMPORTANT: to avoid training on your own test set next run, add this "
          f"line in train.py right after clean_files is built:")
    print(f"    clean_files = clean_files[:-20]  # reserve last 20 for eval")


if __name__ == "__main__":
    main()
