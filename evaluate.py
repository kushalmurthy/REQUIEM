"""
Evaluation harness: runs the model on the held-out eval set (data/eval/,
built by generate_eval_set.py) and scores PESQ/STOI against the matching
clean reference. This produces the baseline-vs-yours table for the PPT.
"""

import glob
import torch
import torchaudio
import numpy as np
from pesq import pesq
from pystoi import stoi

from model import AdaptiveDenoiser
from dataset import SR, N_FFT, HOP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def denoise_file(model, window, noisy_path, sr=SR):
    import soundfile as sf
    audio_np, file_sr = sf.read(noisy_path)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    wav = torch.from_numpy(audio_np).float().unsqueeze(0).to(DEVICE)  # [1, T]
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)

    spec = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)  # [1, F, T]
    mag_in = spec.abs().transpose(1, 2)  # [1, T, F]

    with torch.no_grad():
        mask, _ = model(mag_in)
    mask_bft = mask.transpose(1, 2)  # [1, F, T]

    est_spec = mask_bft * spec
    est_wav = torch.istft(est_spec, N_FFT, HOP, window=window, length=wav.shape[-1])
    return est_wav.squeeze(0).cpu().numpy()


def score_pair(clean_np, enhanced_np, sr=SR):
    min_len = min(len(clean_np), len(enhanced_np))
    clean_np = clean_np[:min_len]
    enhanced_np = enhanced_np[:min_len]
    pesq_score = pesq(sr, clean_np, enhanced_np, 'wb')
    stoi_score = stoi(clean_np, enhanced_np, sr, extended=False)
    return pesq_score, stoi_score


def find_eval_pairs(eval_dir="data/eval"):
    """Auto-discovers (noisy, clean) pairs written by generate_eval_set.py."""
    noisy_files = sorted(glob.glob(f"{eval_dir}/noisy_*.wav"))
    pairs = []
    for noisy_path in noisy_files:
        idx = noisy_path.split("noisy_")[-1]
        clean_path = f"{eval_dir}/clean_{idx}"
        import os
        if os.path.exists(clean_path):
            pairs.append((noisy_path, clean_path))
    return pairs


def evaluate_pairs(model, window, test_pairs):
    import soundfile as sf
    results = []
    for noisy_path, clean_path in test_pairs:
        clean_np, csr = sf.read(clean_path)
        if clean_np.ndim > 1:
            clean_np = clean_np.mean(axis=1)

        enhanced_np = denoise_file(model, window, noisy_path)
        p, s = score_pair(clean_np, enhanced_np)
        results.append((noisy_path, p, s))
        print(f"{noisy_path}: PESQ={p:.3f}  STOI={s:.3f}")

    scores = np.array([(p, s) for _, p, s in results])
    print(f"\nMean PESQ={scores[:, 0].mean():.3f}  Mean STOI={scores[:, 1].mean():.3f}")
    return results


if __name__ == "__main__":
    import sys

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_epoch50.pt"
    print(f"Evaluating checkpoint: {checkpoint}")

    window = torch.hann_window(N_FFT).to(DEVICE)
    model = AdaptiveDenoiser(n_freq=N_FFT // 2 + 1, n_classes=7, use_adaptive=True).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    test_pairs = find_eval_pairs()
    if not test_pairs:
        print("No eval pairs found in data/eval/. Run generate_eval_set.py first.")
    else:
        print(f"Found {len(test_pairs)} eval pairs.")
        evaluate_pairs(model, window, test_pairs)
