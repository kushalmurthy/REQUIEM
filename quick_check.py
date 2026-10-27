import torch, torchaudio, glob
import soundfile as sf

files = glob.glob("data/LibriSpeech/dev-clean/**/*.flac", recursive=True)
audio_np, sr = sf.read(files[0])
wav = torch.from_numpy(audio_np).float().unsqueeze(0)  # [1, T]
print("loaded:", files[0], "sr:", sr, "shape:", wav.shape)

if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000

n_fft, hop = 512, 128
window = torch.hann_window(n_fft)

spec = torch.stft(wav, n_fft, hop, window=window, return_complex=True)
recon = torch.istft(spec, n_fft, hop, window=window, length=wav.shape[-1])

print("max reconstruction error:", (wav - recon).abs().max().item())
sf.write("recon_test.wav", recon.squeeze(0).numpy(), sr)
print("saved recon_test.wav — listen to it")