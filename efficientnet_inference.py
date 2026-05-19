"""
EfficientNet inference helper — runs under WSL Python (srnet_env).
Can be used as a module OR called from the command line:

    python efficientnet_inference.py /mnt/d/path/to/image.jpg

Prints a single JSON line to stdout, e.g.:
    {"is_stego": true, "confidence": 87.34}
"""

import os
import sys
import numpy as np

_APP_DIR = os.path.dirname(os.path.abspath(__file__))

if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

CHECKPOINT_PATH = os.path.join(
    _APP_DIR, "EfficientNet_ALASKA2_8thrun", "ckpt-780000"
)
CROP_SIZE = 512
# JUNIWARD ckpt-90000 was trained with 11 MBConv blocks kept (cutoff block5c,
# trunk output = 112 channels → Dense kernel shape [112, 2] in the checkpoint).
# Keras EfficientNetB0 has 16 MBConv blocks, so remove_n = 16 - 11 = 5.
REMOVE_N  = 7

# ── module-level state (loaded once) ────────────────────────────────────────
_model = None


def _load_model():
    """Build EfficientNetB0 (rm-8, stem-ablated) and restore weights. Called once."""
    global _model

    if _model is not None:
        return _model

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Import build function from training script
    from train_efficientnet import build_efficientnet_model

    # Pass explicit args — training script's current REMOVE_N/CROP_SIZE may differ
    model = build_efficientnet_model(
        input_shape=(CROP_SIZE, CROP_SIZE, 3),
        remove_n=REMOVE_N,
    )

    # Training saved with `Checkpoint(optimizer=optimizer, model=model, step=...)`.
    # Under Keras 3 / TF 2.16+ only the BN moving_mean / moving_variance land
    # under the model/ tree; every trainable (all conv kernels, BN gamma/beta,
    # final Dense) lives under `optimizer/_trainable_variables/{i}`. Restoring
    # `Checkpoint(model=model)` alone silently loads only the BN stats —
    # trainables stay at ImageNet/random init and `expect_partial()` hides it,
    # so predictions end up random and inconsistent across processes. Load both.
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(CHECKPOINT_PATH).expect_partial()

    reader = tf.train.load_checkpoint(CHECKPOINT_PATH)
    loaded = mismatched = missing = 0
    for i, var in enumerate(model.trainable_variables):
        key = f"optimizer/_trainable_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE"
        if not reader.has_tensor(key):
            missing += 1
            continue
        val = reader.get_tensor(key)
        if tuple(val.shape) != tuple(var.shape):
            print(f"  shape mismatch at var {i} ({var.name}): "
                  f"ckpt={tuple(val.shape)} model={tuple(var.shape)}")
            mismatched += 1
            continue
        var.assign(val)
        loaded += 1
    print(f"  restored {loaded}/{len(model.trainable_variables)} trainables "
          f"({mismatched} shape mismatch, {missing} missing in checkpoint)")

    _model = model
    return model


def _preprocess(image_path: str) -> np.ndarray:
    """
    Open image as RGB, resize so shortest side >= CROP_SIZE,
    centre-crop to CROP_SIZE×CROP_SIZE.
    Returns shape (1, 256, 256, 3) float32 in [0, 255].
    EfficientNetB0 rescales internally.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Upscale if either dimension is smaller than crop size
    if w < CROP_SIZE or h < CROP_SIZE:
        scale = max(CROP_SIZE / w, CROP_SIZE / h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        w, h = img.size

    # Centre crop
    left = (w - CROP_SIZE) // 2
    top  = (h - CROP_SIZE) // 2
    img  = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    arr = np.array(img, dtype=np.float32)       # (256, 256, 3), [0, 255]
    arr = arr[np.newaxis]                        # (1, 256, 256, 3)
    return arr


def _d4_orientations(arr: np.ndarray) -> np.ndarray:
    """
    Generate the 8 D4 dihedral orientations of an image:
    {identity, rot90, rot180, rot270} × {no flip, horizontal flip}.

    arr shape: (1, H, W, 3) — output shape: (8, H, W, 3).

    These are the same orientations the training pipeline samples from at
    random, so the model has seen all of them and should give consistent
    predictions; averaging cuts noise.
    """
    img = arr[0]                                                # (H, W, 3)
    out = []
    for flip in (False, True):
        x = np.flip(img, axis=1) if flip else img              # horizontal flip
        for k in range(4):
            out.append(np.rot90(x, k=k, axes=(0, 1)))           # 0/90/180/270°
    return np.stack(out, axis=0).astype(np.float32)             # (8, H, W, 3)


def predict(image_path: str, use_tta: bool = True):
    """
    Run EfficientNet on a single image, with optional test-time augmentation.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    use_tta : bool
        If True (default), run inference on all 8 D4 dihedral orientations
        and average the softmax outputs. Typically adds +1-2 pp accuracy
        at the cost of ~8× inference time. Set False for fastest single-pass
        inference.

    Returns
    -------
    is_stego   : bool  — True if steganography detected
    confidence : float — confidence of predicted class, 0–100 %
    """
    import tensorflow as tf

    model = _load_model()
    arr   = _preprocess(image_path)                              # (1, H, W, 3)

    if use_tta:
        # Process each orientation one at a time and accumulate the softmax.
        # We can't batch 8 on CPU because intermediate BN tensors at 512×512
        # input × stem-stride-1 ablation reach ~300 MB each at batch 8 — OOM
        # on a laptop CPU. Running batch=1 eight times is slower but fits.
        orientations = _d4_orientations(arr)                     # (8, H, W, 3)
        probs_sum = np.zeros(2, dtype=np.float32)
        for i in range(orientations.shape[0]):
            single = orientations[i:i + 1]                       # (1, H, W, 3)
            logits = model(single, training=False)               # (1, 2)
            probs_sum += tf.nn.softmax(logits, axis=-1).numpy()[0]
        probs = probs_sum / orientations.shape[0]                # (2,)
    else:
        logits = model(arr, training=False)                      # (1, 2)
        probs  = tf.nn.softmax(logits, axis=-1).numpy()[0]       # (2,)

    class_id   = int(np.argmax(probs))
    confidence = float(probs[class_id]) * 100.0

    is_stego = (class_id == 1)
    return is_stego, round(confidence, 2)


# ── CLI entry point ──────────────────────────────────────────────────────────
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
