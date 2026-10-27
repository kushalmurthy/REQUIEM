"""
Quick end-to-end training loop test using SYNTHETIC (random) noise instead
of real files. This exists purely to confirm train.py's pipeline works
(shapes, loss, backward pass, optimizer step) without waiting on Person 2's
real noise dataset.

Once real noise files exist in data/noise/<class>/, switch back to train.py
with real file lists -- this script is a smoke test only, not for producing
a real model.

Run: python quick_train_test.py
"""

import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import AdaptiveDenoiser
from losses import combined_loss

SR = 16000
N_FFT = 512
HOP = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SyntheticNoiseDataset(Dataset):
    """Same interface as the real DenoiseDataset, but generates random
    noise instead of loading noise files -- so training can be tested
    before real noise data exists."""

    def __init__(self, clean_files, snr_range=(-5, 15), segment_sec=4.0,
                 sr=SR, n_classes=5):
        self.clean_files = clean_files
        self.snr_range = snr_range
        self.sr = sr
        self.seg_len = int(segment_sec * sr)
        self.n_classes = n_classes

    def __len__(self):
        return len(self.clean_files)

    def _load_segment(self, path):
        import soundfile as sf
        audio_np, sr = sf.read(path)
        wav = torch.from_numpy(audio_np).float()
        if wav.dim() > 1:
            wav = wav.mean(-1)
        if sr != self.sr:
            import torchaudio
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sr).squeeze(0)
        if wav.shape[0] < self.seg_len:
            wav = torch.nn.functional.pad(wav, (0, self.seg_len - wav.shape[0]))
        else:
            start = random.randint(0, wav.shape[0] - self.seg_len)
            wav = wav[start:start + self.seg_len]
        return wav

    def __getitem__(self, idx):
        clean = self._load_segment(self.clean_files[idx])

        # SYNTHETIC noise: just random gaussian noise, shaped like real noise
        # would be. Good enough to test the pipeline, useless for real quality.
        noise = torch.randn(self.seg_len) * 0.3
        label = random.randint(0, self.n_classes - 1)

        snr_db = random.uniform(*self.snr_range)
        clean_power = (clean ** 2).mean() + 1e-8
        noise_power = (noise ** 2).mean() + 1e-8
        scale = torch.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10)))
        noisy = clean + scale * noise

        peak = noisy.abs().max()
        if peak > 1.0:
            noisy = noisy / peak
            clean = clean / peak

        return noisy, clean, label


def collate_fn(batch):
    noisy = torch.stack([b[0] for b in batch])
    clean = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return noisy, clean, labels


def stft_mag(wav, window):
    spec = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)
    return spec.abs(), spec


def istft_from_mask(mask_bft, spec, window, length):
    est_spec = mask_bft * spec
    return torch.istft(est_spec, N_FFT, HOP, window=window, length=length)


def main():
    window = torch.hann_window(N_FFT).to(DEVICE)

    clean_files = glob.glob("data/LibriSpeech/dev-clean/**/*.flac", recursive=True)
    print(f"Found {len(clean_files)} clean files")
    assert len(clean_files) > 0, "No clean files found — check the LibriSpeech download path"

    # use a small subset for a fast smoke test
    clean_files = clean_files[:32]

    dataset = SyntheticNoiseDataset(clean_files, n_classes=5)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = AdaptiveDenoiser(n_freq=N_FFT // 2 + 1, n_classes=5, use_adaptive=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Running 2 test epochs on synthetic noise (smoke test only)...")
    for epoch in range(2):
        model.train()
        total_loss = 0.0
        for noisy, clean, labels in loader:
            noisy, clean, labels = noisy.to(DEVICE), clean.to(DEVICE), labels.to(DEVICE)

            noisy_mag, noisy_spec = stft_mag(noisy, window)
            clean_mag, _ = stft_mag(clean, window)

            mag_in = noisy_mag.transpose(1, 2)
            mask, logits = model(mag_in)
            mask_bft = mask.transpose(1, 2)

            est_wav = istft_from_mask(mask_bft, noisy_spec, window, length=noisy.shape[-1])
            est_mag = mask_bft * noisy_mag

            loss = combined_loss(est_wav, clean, est_mag, clean_mag)
            cls_loss = nn.functional.cross_entropy(logits.mean(dim=1), labels)
            loss = loss + 0.1 * cls_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/2  avg_loss={total_loss / len(loader):.4f}")

    print("\nSUCCESS: training loop ran end-to-end with no crashes.")
    print("This validates the pipeline only -- loss numbers here are meaningless")
    print("since the 'noise' was random static, not real defense-relevant noise.")
    print("Once Person 2's real noise files exist, switch to train.py with real data.")


if __name__ == "__main__":
    main()