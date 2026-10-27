"""
Loss functions. Combined SI-SNR + spectral L1, weighted 0.7/0.3 by default
(tune alpha if output sounds muddy -- raise alpha toward SI-SNR; if it sounds
distorted/artifacty, lower alpha toward spectral).
"""

import torch
import torch.nn.functional as F


def si_snr(estimate, target, eps=1e-8):
    """Scale-invariant SNR in dB. estimate, target: [B, T] time-domain waveforms."""
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    s_target = (torch.sum(estimate * target, dim=-1, keepdim=True) /
                (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)) * target
    e_noise = estimate - s_target

    ratio = torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def si_snr_loss(estimate, target):
    return -si_snr(estimate, target).mean()


def spectral_l1_loss(est_mag, target_mag):
    return F.l1_loss(est_mag, target_mag)


def combined_loss(estimate_wav, target_wav, est_mag, target_mag, alpha=0.7):
    return (alpha * si_snr_loss(estimate_wav, target_wav) +
            (1 - alpha) * spectral_l1_loss(est_mag, target_mag))
