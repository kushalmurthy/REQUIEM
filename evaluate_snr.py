"""
Computes SI-SNR (Scale-Invariant Signal-to-Noise Ratio, in dB) for every
saved checkpoint against the held-out eval set, plus the baseline (no
denoising) SI-SNR for comparison. Higher is better -- this is literally the
metric your training loss optimizes, so it should track your training loss
curve more directly than PESQ/STOI do.

Run once: python evaluate_snr.py
It auto-discovers all checkpoint_epoch*.pt files in the current folder.
"""

import glob
import os
import re
import torch
import soundfile as sf
import numpy as np

from model import AdaptiveDenoiser
from dataset import SR, N_FFT, HOP
from losses import si_snr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_eval_pairs(eval_dir="data/eval"):
    noisy_files = sorted(glob.glob(f"{eval_dir}/noisy_*.wav"))
    pairs = []
    for noisy_path in noisy_files:
        idx = noisy_path.split("noisy_")[-1]
        clean_path = f"{eval_dir}/clean_{idx}"
        if os.path.exists(clean_path):
            pairs.append((noisy_path, clean_path))
    return pairs


def load_wav(path, sr=SR):
    audio_np, file_sr = sf.read(path)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    wav = torch.from_numpy(audio_np).float()
    return wav


def denoise(model, window, noisy_wav):
    wav = noisy_wav.unsqueeze(0).to(DEVICE)
    spec = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)
    mag_in = spec.abs().transpose(1, 2)
    with torch.no_grad():
        mask, _ = model(mag_in)
    mask_bft = mask.transpose(1, 2)
    est_spec = mask_bft * spec
    est_wav = torch.istft(est_spec, N_FFT, HOP, window=window, length=wav.shape[-1])
    return est_wav.squeeze(0)


def compute_baseline_snr(pairs):
    scores = []
    for noisy_path, clean_path in pairs:
        noisy = load_wav(noisy_path)
        clean = load_wav(clean_path)
        min_len = min(len(noisy), len(clean))
        score = si_snr(noisy[:min_len].unsqueeze(0), clean[:min_len].unsqueeze(0)).item()
        scores.append(score)
    return np.mean(scores)


def compute_model_snr(model, window, pairs):
    scores = []
    for noisy_path, clean_path in pairs:
        noisy = load_wav(noisy_path)
        clean = load_wav(clean_path)
        enhanced = denoise(model, window, noisy).cpu()
        min_len = min(len(enhanced), len(clean))
        score = si_snr(enhanced[:min_len].unsqueeze(0), clean[:min_len].unsqueeze(0)).item()
        scores.append(score)
    return np.mean(scores)


if __name__ == "__main__":
    pairs = find_eval_pairs()
    if not pairs:
        print("No eval pairs found in data/eval/. Run generate_eval_set.py first.")
        exit()

    print(f"Found {len(pairs)} eval pairs.\n")

    baseline_snr = compute_baseline_snr(pairs)
    print(f"BASELINE (no denoising) SI-SNR: {baseline_snr:.2f} dB\n")

    window = torch.hann_window(N_FFT).to(DEVICE)

    checkpoints = sorted(
        glob.glob("checkpoint_epoch*.pt"),
        key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))
    )

    if not checkpoints:
        print("No checkpoint_epoch*.pt files found in this folder.")
        exit()

    print(f"{'Epoch':>6} | {'SI-SNR (dB)':>12} | {'Improvement over baseline':>26}")
    print("-" * 50)

    results = []
    for ckpt in checkpoints:
        epoch = int(re.search(r"epoch(\d+)", ckpt).group(1))
        model = AdaptiveDenoiser(n_freq=N_FFT // 2 + 1, n_classes=7, use_adaptive=True).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()

        snr = compute_model_snr(model, window, pairs)
        improvement = snr - baseline_snr
        results.append((epoch, snr))
        print(f"{epoch:>6} | {snr:>10.2f} dB | {improvement:>+22.2f} dB")

    best_epoch, best_snr = max(results, key=lambda x: x[1])
    print(f"\nBest checkpoint by SI-SNR: epoch {best_epoch} ({best_snr:.2f} dB, "
          f"+{best_snr - baseline_snr:.2f} dB over baseline)")
