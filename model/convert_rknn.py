"""
Convert gprnet.onnx to gprnet.rknn for RK3568 NPU deployment.

Must run on x86 Ubuntu Linux with rknn-toolkit2 installed.
  pip install rknn_toolkit2-*.whl   (from rockchip-linux/rknn-toolkit2 GitHub)

Usage:
  python model/convert_rknn.py                          # FP16, no quantization
  python model/convert_rknn.py --quantize               # INT8 quantization
  python model/convert_rknn.py --quantize --dataset model/rknn_dataset.txt

Normalization matches training preprocessing:
  normalized = (pixel/255 - 0.5) / 0.5
  => mean_values = [[127.5]],  std_values = [[127.5]]   (grayscale single channel)
"""

import argparse
import sys
from pathlib import Path

MODEL_DIR   = Path(__file__).parent
ONNX_PATH   = MODEL_DIR / "gprnet.onnx"
RKNN_PATH   = MODEL_DIR / "gprnet.rknn"
DATASET_TXT = MODEL_DIR / "rknn_dataset.txt"


def build_dataset_txt(bscan_dir: Path, out_path: Path, max_images: int = 50) -> None:
    """Write a calibration dataset .txt file (one image path per line)."""
    pngs = sorted(bscan_dir.glob("*.png"))[:max_images]
    if not pngs:
        print(f"WARNING: no PNG images found in {bscan_dir}")
        return
    with open(out_path, "w") as f:
        for p in pngs:
            f.write(str(p) + "\n")
    print(f"Dataset: {len(pngs)} images -> {out_path}")


def convert(onnx_path: Path, rknn_path: Path, quantize: bool,
            dataset_txt: Path, target: str) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)

    rknn.config(
        mean_values=[[127.5]],
        std_values=[[127.5]],
        target_platform=target,
        quant_img_RGB2BGR=False,
    )

    print(f"Loading {onnx_path} ...")
    ret = rknn.load_onnx(
        model=str(onnx_path),
        input_size_list=[[1, 1, 256, 256]],
    )
    if ret != 0:
        print(f"ERROR: load_onnx failed (ret={ret})")
        sys.exit(1)

    print(f"Building (quantize={quantize}) ...")
    if quantize:
        if not dataset_txt.exists():
            print(f"ERROR: dataset file not found: {dataset_txt}")
            print("       Run with --gen-dataset first, or provide --dataset <path>")
            sys.exit(1)
        ret = rknn.build(do_quantization=True, dataset=str(dataset_txt))
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"ERROR: build failed (ret={ret})")
        sys.exit(1)

    print(f"Exporting -> {rknn_path} ...")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        print(f"ERROR: export_rknn failed (ret={ret})")
        sys.exit(1)

    size_mb = rknn_path.stat().st_size / 1e6
    quant_tag = "INT8" if quantize else "FP16"
    print(f"Done: {rknn_path}  ({size_mb:.2f} MB,  {quant_tag},  target={target})")
    rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",        default=str(ONNX_PATH))
    parser.add_argument("--output",      default=str(RKNN_PATH))
    parser.add_argument("--target",      default="rk3568",
                        help="Target SoC: rk3568 / rk3566 / rk3588 / rv1106")
    parser.add_argument("--quantize",    action="store_true",
                        help="INT8 post-training quantization (needs --dataset)")
    parser.add_argument("--dataset",     default=str(DATASET_TXT),
                        help="Path to calibration dataset .txt (one image per line)")
    parser.add_argument("--gen-dataset", action="store_true",
                        help="Auto-generate rknn_dataset.txt from bscan_preview/ images")
    parser.add_argument("--bscan-dir",
                        default=str(MODEL_DIR.parent / "gprmax_sealed_bag_V8" / "bscan_preview"),
                        help="Directory containing B-scan PNG images for calibration")
    args = parser.parse_args()

    onnx_path   = Path(args.onnx)
    rknn_path   = Path(args.output)
    dataset_txt = Path(args.dataset)

    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found: {onnx_path}")
        print("       Run: python model/export_onnx.py")
        sys.exit(1)

    if args.gen_dataset:
        build_dataset_txt(Path(args.bscan_dir), dataset_txt)
        if not args.quantize:
            print("Dataset generated. Add --quantize to use it for INT8 conversion.")
            return

    convert(onnx_path, rknn_path, args.quantize, dataset_txt, args.target)


if __name__ == "__main__":
    main()
