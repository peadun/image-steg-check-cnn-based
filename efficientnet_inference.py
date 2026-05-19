"""
EfficientNet inference helper using ONNX Runtime (no TensorFlow required).
Can be used as a module OR called from the command line:

    python efficientnet_inference.py /path/to/image.jpg

Prints a single JSON line to stdout, e.g.:
    {"is_stego": true, "confidence": 87.34}
"""

import os
import sys
import numpy as np

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

ONNX_PATH  = os.path.join(_APP_DIR, 'model.onnx')
CROP_SIZE  = 512

_session = None


def _load_model():
    global _session
    if _session is not None:
        return _session
    import onnxruntime as rt

    # Memory-conservative session options for low-RAM hosts (e.g. Render free 512 MB).
    # The model has stem stride=1, so early-layer 512×512 activations dominate
    # peak memory. Disabling the memory arena + pattern optimisation trades some
    # speed for ~30-50% lower peak usage.
    opts = rt.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern   = False
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    _session = rt.InferenceSession(ONNX_PATH, sess_options=opts)
    print(f"  ONNX model loaded ({os.path.getsize(ONNX_PATH) / 1024 / 1024:.1f} MB)")
    return _session


def _preprocess(image_path: str) -> np.ndarray:
    """
    Open image as RGB, resize so shortest side >= CROP_SIZE,
    centre-crop to CROP_SIZE×CROP_SIZE.
    Returns shape (1, 512, 512, 3) float32 in [0, 255].
    EfficientNetB0 rescales internally (baked into the ONNX graph).
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if w < CROP_SIZE or h < CROP_SIZE:
        scale = max(CROP_SIZE / w, CROP_SIZE / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        w, h = img.size

    left = (w - CROP_SIZE) // 2
    top  = (h - CROP_SIZE) // 2
    img  = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    arr = np.array(img, dtype=np.float32)
    arr = arr[np.newaxis]                        # (1, 512, 512, 3)
    return arr


def _d4_orientations(arr: np.ndarray) -> np.ndarray:
    """
    Generate the 8 D4 dihedral orientations of an image:
    {identity, rot90, rot180, rot270} × {no flip, horizontal flip}.
    arr shape: (1, H, W, 3) — output shape: (8, H, W, 3).
    """
    img = arr[0]
    out = []
    for flip in (False, True):
        x = np.flip(img, axis=1) if flip else img
        for k in range(4):
            out.append(np.rot90(x, k=k, axes=(0, 1)))
    return np.stack(out, axis=0).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict(image_path: str, use_tta: bool = True):
    """
    Run EfficientNet on a single image, with optional test-time augmentation.

    Parameters
    ----------
    image_path : str
    use_tta    : bool — if True, average predictions over all 8 D4 orientations.

    Returns
    -------
    is_stego   : bool
    confidence : float — 0–100 %
    """
    session    = _load_model()
    input_name = session.get_inputs()[0].name
    arr        = _preprocess(image_path)

    if use_tta:
        orientations = _d4_orientations(arr)
        probs_sum = np.zeros(2, dtype=np.float32)
        for i in range(orientations.shape[0]):
            single = orientations[i:i + 1]
            logits = session.run(None, {input_name: single})[0][0]
            probs_sum += _softmax(logits)
        probs = probs_sum / orientations.shape[0]
    else:
        logits = session.run(None, {input_name: arr})[0][0]
        probs  = _softmax(logits)

    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id]) * 100.0
    is_stego   = (class_id == 1)
    return is_stego, round(confidence, 2)


if __name__ == "__main__":
    import json

    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: efficientnet_inference.py <image_path>"}))
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(json.dumps({"error": f"File not found: {img_path}"}))
        sys.exit(1)

    try:
        is_stego, confidence = predict(img_path)
        print(json.dumps({"is_stego": is_stego, "confidence": confidence}))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
