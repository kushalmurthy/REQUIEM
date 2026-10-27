# Adaptive Noise Cancellation — Model Scaffold

Starter code for the ML side of the project: causal STFT-masking denoiser
with a FiLM-conditioned adaptive mechanism (noise-type classifier steering
the denoiser). Matches the plan: DTLN-style causal masking, SI-SNR + spectral
loss, PESQ/STOI evaluation, INT8 export for on-device deployment.

## Files

- `model.py` — the network (`AdaptiveDenoiser`). Set `use_adaptive=False` to
  get a plain baseline for your "with vs without adaptation" comparison.
- `dataset.py` — on-the-fly clean+noise mixing at random SNR (−5 to +15 dB).
- `losses.py` — SI-SNR (time domain) + spectral L1, combined.
- `train.py` — training loop. Fill in file paths in `main()`.
- `evaluate.py` — PESQ/STOI scoring harness. This produces your PPT table.
- `export_quantize.py` — ONNX export + INT8 quantization for P3.

## Setup

```bash
conda create -n anc python=3.10 -y
conda activate anc
pip install torch torchaudio numpy scipy soundfile librosa
pip install pesq pystoi onnx onnxruntime
```

## Before touching this code

1. **Verify STFT round-trip** on a real audio file first (see the snippet
   from earlier in the conversation). If reconstruction error isn't ~1e-6,
   stop and fix it before training anything.
2. **Get the pretrained DTLN baseline running** and listen to its output.
   This is your reference point for "did fine-tuning actually help."
3. **Confirm the latency/parameter budget with your embedded lead** before
   you scale up `hidden` in `model.py` — bigger hidden size = better
   quality but slower inference. Default here (128) is a reasonable
   starting point for a Raspberry Pi–class board; adjust after you hear back.

## Workflow

```bash
# 1. Fill in clean_files / noise_files / noise_labels in train.py, then:
python train.py

# 2. Fill in test_pairs in evaluate.py, then score:
python evaluate.py

# 3. Once metrics look good, export + quantize for on-device deployment:
python export_quantize.py

# 4. IMPORTANT: re-run evaluate.py against the quantized model
#    (via onnxruntime.InferenceSession) before handing to P3.
#    Quantization can silently drop quality -- catch it here, not on hardware.
```

## Noise class convention

`noise_labels` in `train.py`/`dataset.py` expects integer class ids, e.g.:

```
0 = engine
1 = rotor
2 = gunfire
3 = wind
4 = radio chatter
```

Adjust `n_classes` in `model.py` / `train.py` to match how many categories
P2's dataset actually covers. Keep it small (4-6 classes) — more classes
means more data needed per class to train the classifier reliably in your
timeframe.

## Debugging checklist if metrics aren't improving

- Confirm `clean_files` and `noise_files` aren't empty / paths are correct
  (there's an `assert` for this in `train.py`, but double check contents).
- Check the SI-SNR loss is going *down* — if it's flat or NaN, lower the
  learning rate first (try `5e-4`) before changing architecture.
- Listen to actual output files, not just the metric numbers — over-
  suppression scores deceptively well on some metrics but sounds bad.
- Make sure your train/val split uses entirely separate noise *recordings*,
  not just separate random clips from the same recording (data leak).
