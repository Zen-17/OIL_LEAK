"""
Export GPRNet to ONNX for embedded Linux / RKNN deployment.

Usage:
  conda activate torch_gpu
  cd D:\\GPRMax
  python model/export_onnx.py
  python model/export_onnx.py --output model/gprnet.onnx --opset 17

Outputs:
  model/gprnet.onnx  — dynamic-batch ONNX model (opset 13 by default)

Verify (requires onnxruntime):
  pip install onnxruntime
  python model/export_onnx.py --verify
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from gprnet import GPRNet

MODEL_DIR = Path(__file__).parent
DEVICE    = torch.device("cpu")   # export on CPU for portability


def export(ckpt_path: Path, output_path: Path, opset: int) -> None:
    model = GPRNet().to(DEVICE)
    model.load_state_dict(
        torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    dummy = torch.zeros(1, 1, 256, 256, device=DEVICE)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["bscan"],
        output_names=["logit_a", "logits_b"],
        dynamic_axes={
            "bscan":    {0: "batch"},
            "logit_a":  {0: "batch"},
            "logits_b": {0: "batch"},
        },
        do_constant_folding=True,
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported: {output_path}  ({size_mb:.2f} MB,  opset={opset})")


def verify(output_path: Path) -> None:
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("onnxruntime not installed -- skipping verification.")
        print("  pip install onnxruntime")
        return

    import onnx
    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    print("ONNX graph check: OK")

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    dummy = np.zeros((2, 1, 256, 256), dtype=np.float32)
    logit_a, logits_b = sess.run(None, {"bscan": dummy})
    print(f"ORT inference: OK  logit_a={logit_a.shape}  logits_b={logits_b.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default=str(MODEL_DIR / "gprnet_dual.pth"))
    parser.add_argument("--output", default=str(MODEL_DIR / "gprnet.onnx"))
    parser.add_argument("--opset",  type=int, default=13,
                        help="ONNX opset version (13=RKNN-compatible, 17=latest)")
    parser.add_argument("--verify", action="store_true",
                        help="Run onnxruntime inference check after export")
    args = parser.parse_args()

    ckpt_path   = Path(args.ckpt)
    output_path = Path(args.output)

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    export(ckpt_path, output_path, args.opset)

    if args.verify:
        verify(output_path)


if __name__ == "__main__":
    main()
