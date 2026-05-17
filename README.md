# JPEG Steganography Detection

A lightweight CNN-based steganalysis tool for detecting hidden data in JPEG
images. Built around an EfficientNetB0 backbone with steganalysis-specific
architectural modifications following Hong et al. (2023). Targets three
adaptive JPEG-domain steganographic algorithms: **JUNIWARD**, **JMiPOD**,
and **UERD**.

The project ships three user-facing interfaces:

- A **Flask web application** with drag-and-drop upload
- A **Firefox browser extension** with a right-click "Check for steganography" context menu
- A **Tkinter desktop prototype**

## Results

Evaluated on a 22,500-pair held-out test split of the ALASKA2 dataset
(three-algorithm mixed, single seed):

| Configuration | Test accuracy | Probability of Error |
|---|---|---|
| Single model (baseline) | 75.91% | 0.2409 |
| Single model + TTA | 76.22% | 0.2378 |
| 5-model snapshot ensemble | 75.91% | 0.2409 |
| 5-model ensemble + TTA | **76.28%** | **0.2372** |

The deployed inference path defaults to single model + 8-orientation D4 test-time
augmentation (TTA) for a fast-but-accurate balance. The 5-model snapshot
ensemble produced no additional measurable gain — see the discussion below.

## Architecture

Following Hong et al. (2023), the model is **not** a generic EfficientNetB0 —
two steganalysis-specific modifications are applied:

1. **Stem stride ablation** — first Conv2D stride is reduced from 2 to 1,
   preserving the four-times-larger feature map after the stem so that the
   high-frequency components carrying steganographic perturbations are
   retained rather than discarded by aggressive downsampling.
2. **Block-wise pruning to 9 MBConv blocks** (paper's "rm-8" configuration
   under TF Keras counting, `REMOVE_N = 7`) — drops the deep semantic-feature
   blocks that contribute to object recognition but are irrelevant to
   steganalysis. The remaining trunk concentrates capacity in the early
   layers where local fine-grained features dominate.

Total parameters: **445,233** (434,890 trainable, 10,343 non-trainable).
Input shape: 512×512 RGB. Output: Dense(2) logits where class 0 = cover,
class 1 = stego.

The training pipeline applies the full D4 dihedral augmentation set
(8 orientations — 4 rotations × 2 horizontal-flip states) identically to
the cover and stego image of each pair.

## Scope

This tool is **specialised to JPEG-domain adaptive steganography** (JUNIWARD,
JMiPOD, UERD). It will **not** reliably detect:

- Spatial-domain LSB steganography on PNG/BMP images.
- nsF5, HUGO, or other JPEG algorithms not in the training set.
- Substantially out-of-distribution JPEGs (e.g., images re-compressed many
  times by social-media platforms with different quality factors than the
  ALASKA2 training distribution).

The web app and browser extension only accept `.jpg` / `.jpeg` files to
enforce this specialisation at the input layer rather than relying on users
to read the README.

## Setup

### Requirements

- Python 3.10+
- TensorFlow 2.16 or newer (Keras 3)
- Pillow, NumPy, Flask

```bash
pip install -r requirements.txt
```

Inference runs on CPU. Training was conducted on a single NVIDIA RTX 4090
(24 GB VRAM) at batch size 8.

### Dataset

The model was trained on the [ALASKA2 dataset](https://www.kaggle.com/c/alaska2-image-steganalysis)
from the 2020 Kaggle competition. To reproduce training, download the dataset
and update the `ALASKA2_ROOT` path in `train_efficientnet.py` to point at the
local folder.

Expected layout:

```
ALASKA2/
├── Cover/      # 75,000 cover JPEGs
├── JMiPOD/     # 75,000 stego JPEGs (JMiPOD algorithm)
├── JUNIWARD/   # 75,000 stego JPEGs (JUNIWARD algorithm)
└── UERD/       # 75,000 stego JPEGs (UERD algorithm)
```

### Trained checkpoint

The pre-trained checkpoint achieving 76.28% test accuracy with ensemble + TTA
is included in `EfficientNet_ALASKA2_8thrun/` (the `ckpt-780000` files).
Update `CHECKPOINT_PATH` in `efficientnet_inference.py` if you place it
elsewhere.

## Usage

### Web application

```bash
python server.py
```

Then visit `http://localhost:5000` and drag-drop a JPEG image.

The server includes input-validation defences against arbitrary file
uploads, content-type spoofing, decompression bombs, and large-payload
DoS — see the source of `server.py` for details.

### Command-line inference

```bash
python efficientnet_inference.py path/to/image.jpg
```

Output is a single JSON line, e.g.:

```json
{"is_stego": true, "confidence": 87.34}
```

Test-time augmentation is enabled by default and adds ~1-2 pp of accuracy at
the cost of running the model on 8 orientations per image (~6-8 seconds on
CPU). Pass `use_tta=False` to the `predict()` function for a single
forward pass.

### Desktop prototype (Windows)

```bash
python prototype.py
```

Launches a Tkinter window with file picker, image preview, and analysis
button. The prototype invokes `efficientnet_inference.py` via a WSL
subprocess, so WSL with the trained model accessible is required.

### Firefox browser extension

1. Start the Flask server (`python server.py`).
2. Open Firefox and navigate to `about:debugging#/runtime/this-firefox`.
3. Click **"Load Temporary Add-on..."** and select `extension/manifest.json`.
4. Right-click any JPEG image on any webpage → **"Check for steganography"**.

The classification result appears as a Firefox notification. The extension
forwards the right-clicked image's bytes to the local Flask server and
displays the returned verdict.

To install the extension permanently (instead of as a temporary add-on),
submit `extension/` to [addons.mozilla.org](https://addons.mozilla.org/)
as an Unlisted/Self-Distributed extension, retrieve the signed `.xpi`, and
install via `about:addons → ⚙ → Install Add-on From File`.

## Training (optional)

If you wish to retrain or fine-tune the model from scratch on the ALASKA2
dataset:

```bash
python train_efficientnet.py
```

Configuration is at the top of the script (`MAX_ITER`, `TRAIN_BATCH_SIZE`,
`LR_VALUES`, etc.). Training the headline model took approximately 50 hours
on a single RTX 4090.

To evaluate a trained checkpoint on the held-out test set:

```bash
python train_efficientnet.py --mode test \
    --checkpoint EfficientNet_ALASKA2_8thrun/ckpt-780000
```

For ensemble or TTA evaluation:

```bash
python train_efficientnet.py --mode test_ensemble --tta \
    --checkpoints EfficientNet_ALASKA2_8thrun/ckpt-780000
```

## References

- Hong, J., Kang, J., Choi, J., & Hong, J. (2023). Lightweight image
  steganalysis with block-wise pruning. *Scientific Reports*, 13, 16469.
  https://doi.org/10.1038/s41598-023-43386-2

- Cogranne, R., Giboulot, Q., & Bas, P. (2020). ALASKA-2: Challenging
  academic research on steganalysis with realistic images. *IEEE
  International Workshop on Information Forensics and Security (WIFS)*.

- Boroumand, M., Chen, M., & Fridrich, J. (2019). Deep residual network
  for steganalysis of digital images (SRNet). *IEEE Transactions on
  Information Forensics and Security*, 14(5), 1181-1193.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This project was developed as a final-year project. The architecture
reproduces and adapts the design from Hong et al. (2023); the dataset is
provided by the organisers of the ALASKA2 Kaggle competition.
