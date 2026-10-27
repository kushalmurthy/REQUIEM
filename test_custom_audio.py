"""
Run the trained model on ANY audio file you provide -- not just the
held-out eval set. Useful for testing real-world recordings, audio you
record yourself, or clips you find online.

Usage:
    python test_custom_audio.py path/to/your_audio.wav
    python test_custom_audio.py path/to/your_audio.wav checkpoint_epoch45.pt

Output is saved next to this script as <original_name>_enhanced.wav
"""

import sys
import os
import torch
import soundfile as sf
import numpy as np

from model import AdaptiveDenoiser
from dataset import SR, N_FFT, HOP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_prepare(path, target_sr=SR):
    audio_np, file_sr = sf.read(path)

    # force mono if stereo
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    wav = torch.from_numpy(audio_np).float()

    # resample if the file isn't already 16kHz -- your recording almost
    # certainly won't be, so this step matters
    if file_sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav.unsqueeze(0), file_sr, target_sr).squeeze(0)
        print(f"Resampled from {file_sr} Hz to {target_sr} Hz")

    # normalize peak to avoid clipping, same convention as training data
    peak = wav.abs().max()
    if peak > 1.0:
        wav = wav / peak

    return wav


def denoise(model, window, wav):
    wav = wav.unsqueeze(0).to(DEVICE)  # [1, T]
    spec = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)
    mag_in = spec.abs().transpose(1, 2)

    with torch.no_grad():
        mask, logits = model(mag_in)
    mask_bft = mask.transpose(1, 2)

    est_spec = mask_bft * spec
    est_wav = torch.istft(est_spec, N_FFT, HOP, window=window, length=wav.shape[-1])

    # also report which noise class the model detected, if adaptive
    detected_class = None
    if logits is not None:
        probs = torch.softmax(logits.mean(dim=1), dim=-1)
        detected_class = probs.argmax(dim=-1).item()

    return est_wav.squeeze(0).cpu(), detected_class


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_custom_audio.py <path_to_audio.wav> [checkpoint.pt]")
        sys.exit(1)

    input_path = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "checkpoint_epoch45.pt"

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)
    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    print(f"Loading audio: {input_path}")
    wav = load_and_prepare(input_path)
    print(f"Duration: {len(wav) / SR:.2f}s")

    window = torch.hann_window(N_FFT).to(DEVICE)
    model = AdaptiveDenoiser(n_freq=N_FFT // 2 + 1, n_classes=7, use_adaptive=True).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    print(f"Running inference with {checkpoint}...")
    enhanced, detected_class = denoise(model, window, wav)

    from mad_loader import MAD_CLASS_NAMES
    if detected_class is not None:
        print(f"Detected noise type: {MAD_CLASS_NAMES.get(detected_class, 'unknown')} "
              f"(class {detected_class})")

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = f"{base}_enhanced.wav"
    sf.write(out_path, enhanced.numpy(), SR)
    print(f"\nSaved: {out_path}")
    print(f"Compare it against the original ({input_path}) by listening to both.")
