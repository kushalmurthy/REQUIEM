"""
Dynamic mixing dataset: mixes clean speech + noise at a random SNR every time
a sample is drawn, rather than pre-rendering a fixed dataset. This is what
P2 should hand file lists into.

Directory-independent: just give it lists of file paths.
"""

import random
import torch
import torchaudio
from torch.utils.data import Dataset

SR = 16000
N_FFT = 512
HOP = 128


class DenoiseDataset(Dataset):
    def __init__(self, clean_files, noise_files, noise_labels=None,
                 snr_range=(-5, 15), segment_sec=4.0, sr=SR, clean_only_prob=0.05):
        """
        clean_files: list[str] paths to clean speech wavs
        noise_files: list[str] paths to noise wavs
        noise_labels: list[int] class id per noise file (0..n_classes-1).
                      Required only if you're using the adaptive/classifier branch.
        snr_range: (min_db, max_db) to sample mixing SNR from
        clean_only_prob: fraction of samples with NO noise added, so the model
                         learns not to over-suppress clean speech
        """
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.noise_labels = noise_labels
        self.snr_range = snr_range
        self.sr = sr
        self.seg_len = int(segment_sec * sr)
        self.clean_only_prob = clean_only_prob

    def __len__(self):
        return len(self.clean_files)

    def _load_segment(self, path):
        import soundfile as sf
        audio_np, sr = sf.read(path)
        wav = torch.from_numpy(audio_np).float()
        if wav.dim() > 1:
            wav = wav.mean(-1)  # force mono if stereo
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, self.sr).squeeze(0)
        if wav.shape[0] < self.seg_len:
            wav = torch.nn.functional.pad(wav, (0, self.seg_len - wav.shape[0]))
        else:
            start = random.randint(0, wav.shape[0] - self.seg_len)
            wav = wav[start:start + self.seg_len]
        return wav

    def __getitem__(self, idx):
        clean = self._load_segment(self.clean_files[idx])

        noise_idx = random.randint(0, len(self.noise_files) - 1)
        noise = self._load_segment(self.noise_files[noise_idx])
        label = self.noise_labels[noise_idx] if self.noise_labels else 0

        if random.random() < self.clean_only_prob:
            noisy = clean.clone()
            label = -1  # sentinel: "no noise" — excluded from classifier loss
        else:
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
