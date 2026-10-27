import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import AdaptiveDenoiser
from dataset import DenoiseDataset, collate_fn, SR, N_FFT, HOP
from losses import combined_loss
from mad_loader import load_mad_files

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def stft_mag(wav, window):
    spec = torch.stft(wav, N_FFT, HOP, window=window, return_complex=True)  # [B, F, T]
    mag = spec.abs()
    return mag, spec


def istft_from_mask(mask_bft, spec, window, length):
    est_spec = mask_bft * spec
    wav = torch.istft(est_spec, N_FFT, HOP, window=window, length=length)
    return wav


def train_one_epoch(model, loader, optimizer, window):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for batch_idx, (noisy, clean, labels) in enumerate(loader):
        t0 = time.time()
        noisy, clean, labels = noisy.to(DEVICE), clean.to(DEVICE), labels.to(DEVICE)

        noisy_mag, noisy_spec = stft_mag(noisy, window)
        clean_mag, _ = stft_mag(clean, window)

        mag_in = noisy_mag.transpose(1, 2)
        mask, logits = model(mag_in)
        mask_bft = mask.transpose(1, 2)

        est_wav = istft_from_mask(mask_bft, noisy_spec, window, length=noisy.shape[-1])
        est_mag = mask_bft * noisy_mag

        loss = combined_loss(est_wav, clean, est_mag, clean_mag)

        if model.use_adaptive and (labels >= 0).any():
            valid = labels >= 0
            cls_loss = nn.functional.cross_entropy(logits.mean(dim=1)[valid], labels[valid])
            loss = loss + 0.1 * cls_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

        batch_time = time.time() - t0
        print(f"  batch {batch_idx + 1}/{n_batches}  loss={loss.item():.4f}  ({batch_time:.2f}s/batch)")

    return total_loss / n_batches


def main():
    window = torch.hann_window(N_FFT).to(DEVICE)

    clean_files = glob.glob("data/LibriSpeech/dev-clean/**/*.flac", recursive=True)
    noise_files, noise_labels = load_mad_files("datalabeltrain.csv", "data/MAD/MAD_dataset")

    assert len(clean_files) > 0, "clean_files is empty — add your dataset paths"
    assert len(noise_files) == len(noise_labels), "noise_files and noise_labels must match length"

    # --- Set to True for a fast ~1 minute sanity check, False for the real run ---
    SMOKE_TEST = False

    if SMOKE_TEST:
        clean_files = clean_files[:40]
        print(f"SMOKE TEST MODE: using only {len(clean_files)} clean files "
              f"for a fast sanity check. Set SMOKE_TEST=False for real training.")
    # -------------------------------------------------------------------------

    train_set = DenoiseDataset(clean_files, noise_files, noise_labels,
                                segment_sec=2.0 if SMOKE_TEST else 4.0)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)

    n_classes = len(set(noise_labels))
    model = AdaptiveDenoiser(n_freq=N_FFT // 2 + 1, n_classes=n_classes,
                              use_adaptive=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 50
    print(f"Starting training: {len(clean_files)} clean files, "
          f"{len(train_loader)} batches/epoch, device={DEVICE}")

    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        avg_loss = train_one_epoch(model, train_loader, optimizer, window)
        print(f"Epoch {epoch + 1}/{n_epochs} DONE  avg_loss={avg_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch + 1}.pt")
            print(f"  saved checkpoint_epoch{epoch + 1}.pt")


if __name__ == "__main__":
    main()
