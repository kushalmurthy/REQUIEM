"""
Generates denoised audio from your held-out eval set so you can actually
LISTEN to output quality, not just read metric numbers. Run this anytime
you have a checkpoint you want to sanity-check by ear.

Usage: python listen_test.py checkpoint_epoch35.pt
"""

import sys
import os
import torch
import soundfile as sf

from model import AdaptiveDenoiser
from dataset import SR, N_FFT, HOP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "listen_samples"


def denoise_file(model, window, noisy_path, sr=SR):
    audio_np, file_sr = sf.read(noisy_path)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    wav = torch.from_numpy(audio_np).float().unsqueeze(0).to(DEVICE)

    spec = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)
    mag_in = spec.abs().transpose(1, 2)

    with torch.no_grad():
        mask, _ = model(mag_in)
    mask_bft = mask.transpose(1, 2)

    est_spec = mask_bft * spec
    est_wav = torch.istft(est_spec, N_FFT, HOP, window=window, length=wav.shape[-1])
    return est_wav.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoint_epoch35.pt"
    os.makedirs(OUT_DIR, exist_ok=True)

    window = torch.hann_window(N_FFT).to(DEVICE)
    model = AdaptiveDenoiser(n_freq=N_FFT // 2 + 1, n_classes=7, use_adaptive=True).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    # process the first 5 eval pairs so you have a quick set to listen to
    for i in range(5):
        noisy_path = f"data/eval/noisy_{i:02d}.wav"
        if not os.path.exists(noisy_path):
            continue
        enhanced = denoise_file(model, window, noisy_path)
        out_path = os.path.join(OUT_DIR, f"enhanced_{i:02d}.wav")
        sf.write(out_path, enhanced, SR)
        print(f"Saved {out_path}  (compare against {noisy_path} and data/eval/clean_{i:02d}.wav)")

    print(f"\nListen to all three versions of each: noisy (data/eval/), "
          f"enhanced ({OUT_DIR}/), and clean (data/eval/) to judge quality by ear.")
