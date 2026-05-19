"""
Flask server for steganography detection.
Run under WSL Python (srnet_env):
    python server.py

Then open http://localhost:5000 in your browser.

Security hardening — file upload attack surface:
  A1. Path traversal — the user-supplied filename is never used in the
      saved file path. Saved files use a server-generated UUID; only the
      extension is taken from the user (after whitelisting).
  A2. Content-type spoofing — magic bytes are verified against the
      claimed extension before saving. A file claiming to be ".jpg" but
      whose bytes are not a JPEG header is rejected.
  A3. Bandwidth DoS — Flask's MAX_CONTENT_LENGTH caps requests at 20 MB;
      413 responses are returned cleanly via an error handler.
  A4. Decompression bomb — image dimensions are validated *after* PIL's
      lazy header parse but *before* full decode + inference. Images
      claiming dimensions above MAX_IMAGE_DIMENSION are rejected without
      a full RAM-blowing decompression.
"""

import os
import sys
import tempfile
import uuid

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Make sure we can import the inference modules
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import efficientnet_inference

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024   # 20 MB upload limit (A3)

# ── File-upload validation ───────────────────────────────────────────────────
# JPEG only: the model is trained on JPEG (ALASKA2 with JUNIWARD/JMiPOD/UERD,
# all DCT-domain algorithms). PNGs would be out-of-distribution and any
# prediction on them would be meaningless — rejecting them at the input layer
# is both more honest about the tool's specialisation and a smaller attack
# surface.
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg'}
MAX_IMAGE_DIMENSION = 4096   # pixels per side; rejects bomb headers (A4)

# First-bytes signature used to verify uploaded content actually matches the
# claimed extension. JPEG starts with FF D8 FF.
_MAGIC_BYTES = {
    '.jpg':  (b'\xff\xd8\xff',),
    '.jpeg': (b'\xff\xd8\xff',),
}


def _validate_upload(file_storage):
    """
    Returns (ok, ext_or_error). On success, the second element is the lowercase
    extension (e.g. ".jpg"). On failure, it is a human-readable error string.

    The user's filename is NOT reused in any path. We only consume its
    extension after validating it against a whitelist + magic-byte check.
    """
    if not file_storage.filename:
        return False, 'Empty filename'

    ext = os.path.splitext(file_storage.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (f'File type not allowed (got {ext!r}; '
                       f'this tool only accepts JPEG: .jpg or .jpeg)')

    # Verify magic bytes against the declared extension. (A2)
    head = file_storage.stream.read(8)
    file_storage.stream.seek(0)
    if not any(head.startswith(magic) for magic in _MAGIC_BYTES[ext]):
        return False, 'File contents do not match its declared extension'

    return True, ext


@app.errorhandler(413)
def _payload_too_large(_):
    """Friendly JSON error when the 20 MB limit is exceeded (A3)."""
    return jsonify({'error': 'File too large (max 20 MB)'}), 413


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    ok, ext_or_error = _validate_upload(file)
    if not ok:
        return jsonify({'error': ext_or_error}), 400
    ext = ext_or_error

    # Save with a server-generated UUID; the user's filename is never used
    # in the path. (A1: path traversal mitigation.)
    tmp_path = os.path.join(
        tempfile.gettempdir(), f'upload_{uuid.uuid4().hex}{ext}'
    )
    file.save(tmp_path)

    try:
        # Decompression-bomb check — Image.open() is lazy and reads only the
        # header, so we can inspect declared size before the full decode that
        # would actually allocate the pixel buffer. (A4)
        from PIL import Image
        with Image.open(tmp_path) as img:
            w, h = img.size
            if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
                return jsonify({
                    'error': f'Image too large ({w}×{h}; '
                             f'max {MAX_IMAGE_DIMENSION}×{MAX_IMAGE_DIMENSION})'
                }), 400

        use_tta = request.form.get('use_tta', 'true').lower() != 'false'
        is_stego, confidence = efficientnet_inference.predict(tmp_path, use_tta=use_tta)
        return jsonify({
            'model':      'EfficientNet',
            'is_stego':   bool(is_stego),
            'confidence': confidence,
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == '__main__':
    # Load model once at startup so the first request isn't slow
    print('Loading EfficientNet …')
    efficientnet_inference._load_model()
    print('Model ready. Starting server on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
