"""
Adaptive noise cancellation model.

Architecture: causal GRU-based mask predictor operating on STFT magnitude.
The "adaptive" part is a small noise-type classifier whose output conditions
the main denoising GRU via FiLM (feature-wise linear modulation), so the
model behaves differently depending on the detected noise type.

Input/output convention: [Batch, Time, Freq] for all tensors going in/out
of this module (transpose from torch.stft's [B, F, T] before calling).
"""

import torch
import torch.nn as nn


class NoiseClassifier(nn.Module):
    """Lightweight causal classifier: identifies dominant noise type per frame."""

    def __init__(self, n_freq=257, n_classes=5, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_freq, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, mag):
        # mag: [B, T, F]
        out, _ = self.gru(mag)
        logits = self.fc(out)  # [B, T, n_classes]
        return logits


class AdaptiveDenoiser(nn.Module):
    """
    Causal STFT-magnitude masking denoiser with optional noise-type conditioning.

    Set use_adaptive=False to get a plain (non-adaptive) baseline you can
    compare against in your PPT's "with vs without adaptation" table.
    """

    def __init__(self, n_freq=257, hidden=128, n_classes=5, use_adaptive=True):
        super().__init__()
        self.use_adaptive = use_adaptive
        self.n_classes = n_classes

        if use_adaptive:
            self.classifier = NoiseClassifier(n_freq, n_classes, hidden=64)
            self.film = nn.Linear(n_classes, hidden * 2)  # predicts gamma, beta

        self.gru1 = nn.GRU(n_freq, hidden, batch_first=True)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.fc_mask = nn.Linear(hidden, n_freq)

    def forward(self, mag):
        # mag: [B, T, F] linear magnitude spectrogram (NOT log/dB)
        x, _ = self.gru1(mag)

        logits = None
        if self.use_adaptive:
            logits = self.classifier(mag)                # [B, T, n_classes]
            probs = torch.softmax(logits, dim=-1)
            film_params = self.film(probs)                # [B, T, 2*hidden]
            gamma, beta = film_params.chunk(2, dim=-1)
            x = x * (1 + gamma) + beta

        x, _ = self.gru2(x)
        mask = torch.sigmoid(self.fc_mask(x))              # [B, T, F], bounded (0,1)
        return mask, logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # sanity check
    m = AdaptiveDenoiser(n_freq=257, use_adaptive=True)
    dummy = torch.randn(2, 100, 257)  # [B, T, F]
    mask, logits = m(dummy)
    print("mask shape:", mask.shape)      # expect [2, 100, 257]
    print("logits shape:", logits.shape)  # expect [2, 100, 5]
    print("param count:", count_params(m))
