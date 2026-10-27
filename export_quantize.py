"""
Export the trained model to ONNX, then apply dynamic INT8 quantization.
Hand the .onnx (or quantized .onnx) file to P3 for on-device deployment.

Always re-run evaluate.py on the QUANTIZED model before handing it off --
quantization can quietly drop PESQ/STOI, and you want to know before P3
integrates it, not after.
"""

import torch
from model import AdaptiveDenoiser
from dataset import N_FFT

DEVICE = "cpu"  # export on CPU for portability


def export_onnx(checkpoint_path, out_path="denoiser.onnx", n_classes=5, seq_len=100):
    model = AdaptiveDenoiser(n_freq=N_FFT // 2 + 1, n_classes=n_classes, use_adaptive=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, seq_len, N_FFT // 2 + 1)  # [B, T, F]

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["mag"],
        output_names=["mask", "logits"],
        dynamic_axes={"mag": {1: "time"}, "mask": {1: "time"}, "logits": {1: "time"}},
        opset_version=13,
    )
    print(f"Exported to {out_path}")


def quantize_onnx(onnx_path="denoiser.onnx", out_path="denoiser_int8.onnx"):
    # requires: pip install onnxruntime onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=onnx_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {out_path}")


if __name__ == "__main__":
    export_onnx("checkpoint_epoch50.pt")
    quantize_onnx()
    print("\nNext step: re-run evaluate.py against denoiser_int8.onnx "
          "(via onnxruntime.InferenceSession) to confirm PESQ/STOI didn't collapse.")
