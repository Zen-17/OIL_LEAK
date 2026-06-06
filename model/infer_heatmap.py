"""
GPRNet heatmap inference for B-scan anomaly detection.

Sliding-window inference across a B-scan image:
  - Window width: 80 columns, step: 20 columns
  - Each window resized to 256x256 and fed to GPRNet (ONNX)
  - Head-A probability mapped to a JET colormap heatmap overlay
  - Spatial consistency check (ring buffer K=8) for high-confidence alarm

Usage:
  pip install onnxruntime opencv-python pillow numpy
  python model/infer_heatmap.py --image path/to/bscan.png
  python model/infer_heatmap.py --image path/to/bscan.png --save out.png --no-show
  python model/infer_heatmap.py --image path/to/bscan.png --threshold 0.65
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_DIR  = Path(__file__).parent
ONNX_PATH  = MODEL_DIR / "gprnet.onnx"

WIN_W      = 80     # sliding window width (columns)
WIN_STEP   = 20     # step between windows
IMG_SIZE   = 256    # model input size
MEAN, STD  = 0.5, 0.5

# Spatial consistency buffer
RING_K          = 8    # ring buffer length (frames)
PERSIST_MIN     = 5    # frames with prob > threshold to trigger alarm
OIL_RATIO_MIN   = 0.6  # fraction of those frames where head-B = oil


class GPRInference:
    """
    Stateful inference object — maintains the spatial-consistency ring buffer
    across successive B-scan frames.
    """

    def __init__(self, onnx_path: Path, threshold: float = 0.65):
        import onnxruntime as ort
        self.sess      = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold
        self._ring: deque = deque(maxlen=RING_K)

    def _preprocess_window(self, gray_win: np.ndarray) -> np.ndarray:
        """gray_win: (H, W) uint8  ->  (1, 1, 256, 256) float32 tensor."""
        pil = Image.fromarray(gray_win, mode="L").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        arr = np.array(pil, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return arr[np.newaxis, np.newaxis, :, :]

    def _infer_window(self, gray_win: np.ndarray):
        """Returns (prob_a: float, type_b: int)."""
        inp = self._preprocess_window(gray_win)
        logit_a, logits_b = self.sess.run(None, {"bscan": inp})
        prob_a = float(1.0 / (1.0 + np.exp(-logit_a[0])))
        type_b = int(np.argmax(logits_b[0]))
        return prob_a, type_b

    def process_frame(self, gray: np.ndarray):
        """
        Run sliding-window inference on one B-scan frame.

        Parameters
        ----------
        gray : (H, W) uint8 grayscale B-scan

        Returns
        -------
        overlay : (H, W, 3) uint8  BGR heatmap overlaid on the B-scan
        result  : dict with keys:
                    window_probs  list[float]
                    max_prob      float
                    max_col       int
                    type_b        int
                    alarm         bool
        """
        H, W = gray.shape
        windows = list(range(0, W - WIN_W + 1, WIN_STEP))
        if not windows:
            windows = [0]

        probs   = []
        types   = []
        centres = []
        for x in windows:
            col_end = min(x + WIN_W, W)
            win     = gray[:, x:col_end]
            p, t    = self._infer_window(win)
            probs.append(p)
            types.append(t)
            centres.append(x + (col_end - x) // 2)

        col_prob = np.zeros(W, dtype=np.float32)
        for i, cx in enumerate(centres):
            lo = centres[i - 1] // 2 + cx // 2 if i > 0 else 0
            hi = cx // 2 + centres[i + 1] // 2 if i < len(centres) - 1 else W
            col_prob[lo:hi] = probs[i]

        heat_1d = (np.clip(col_prob, 0, 1) * 255).astype(np.uint8)
        heat_1d = heat_1d[np.newaxis, :].repeat(H, axis=0)
        heat_bgr = cv2.applyColorMap(heat_1d, cv2.COLORMAP_JET)

        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay  = cv2.addWeighted(gray_bgr, 0.6, heat_bgr, 0.4, 0)

        best_i    = int(np.argmax(probs))
        best_cx   = centres[best_i]
        max_prob  = probs[best_i]
        max_type  = types[best_i]

        cv2.line(overlay, (best_cx, 0), (best_cx, H - 1), (0, 0, 255), 2)
        label = f"p={max_prob:.2f}"
        cv2.putText(overlay, label,
                    (max(best_cx - 30, 0), 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self._ring.append((max_prob, max_type))
        alarm = self._check_alarm()
        if alarm:
            cv2.putText(overlay, "!! LEAK ALARM !!",
                        (10, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        result = dict(
            window_probs=probs,
            max_prob=max_prob,
            max_col=best_cx,
            type_b=max_type,
            alarm=alarm,
        )
        return overlay, result

    def _check_alarm(self) -> bool:
        if len(self._ring) < RING_K:
            return False
        high = [(p, t) for p, t in self._ring if p > self.threshold]
        if len(high) < PERSIST_MIN:
            return False
        oil_ratio = sum(1 for _, t in high if t == 0) / len(high)
        return oil_ratio >= OIL_RATIO_MIN


CLASS_NAMES = ["oil", "metal_pipe", "nonmetal_pipe", "rock", "background"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True, help="Path to B-scan PNG/JPG")
    parser.add_argument("--onnx",      default=str(ONNX_PATH))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--save",      default="", help="Save overlay to this path")
    parser.add_argument("--no-show",   action="store_true")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}")
        sys.exit(1)
    if not Path(args.onnx).exists():
        print(f"ERROR: ONNX model not found: {args.onnx}")
        print("       Run: python model/export_onnx.py")
        sys.exit(1)

    gray = np.array(Image.open(img_path).convert("L"))
    print(f"B-scan shape : {gray.shape}")

    engine  = GPRInference(Path(args.onnx), threshold=args.threshold)
    overlay, result = engine.process_frame(gray)

    print(f"Windows      : {len(result['window_probs'])}")
    print(f"Max prob     : {result['max_prob']:.3f}  at col {result['max_col']}")
    print(f"Head-B type  : {CLASS_NAMES[result['type_b']]} ({result['type_b']})")
    print(f"Alarm        : {result['alarm']}")

    if args.save:
        cv2.imwrite(args.save, overlay)
        print(f"Saved        : {args.save}")

    if not args.no_show:
        cv2.imshow("GPR Heatmap", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
