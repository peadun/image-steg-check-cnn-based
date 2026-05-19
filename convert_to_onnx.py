"""
One-time conversion of the TF checkpoint to ONNX format.
Run this in WSL with the srnet_env virtualenv active:

    source ~/srnet_env/bin/activate
    pip install tf2onnx
    python3 convert_to_onnx.py

Produces model.onnx in the same directory (~2-4 MB).
After this, inference no longer needs TensorFlow.
"""
import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

CHECKPOINT_PATH = os.path.join(_APP_DIR, "EfficientNet_ALASKA2_8thrun", "ckpt-780000")
CROP_SIZE = 512
REMOVE_N  = 7

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from train_efficientnet import build_efficientnet_model

print("Building model...")
model = build_efficientnet_model(
    input_shape=(CROP_SIZE, CROP_SIZE, 3),
    remove_n=REMOVE_N,
)

print("Restoring checkpoint...")
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(CHECKPOINT_PATH).expect_partial()

reader = tf.train.load_checkpoint(CHECKPOINT_PATH)
loaded = missing = 0
for i, var in enumerate(model.trainable_variables):
    key = f"optimizer/_trainable_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE"
    if not reader.has_tensor(key):
        missing += 1
        continue
    var.assign(reader.get_tensor(key))
    loaded += 1
print(f"  restored {loaded}/{len(model.trainable_variables)} trainables ({missing} missing)")

print("Converting to ONNX (this may take a minute)...")
import tf2onnx

input_signature = [tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)

output_path = os.path.join(_APP_DIR, 'model.onnx')
with open(output_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"Saved to {output_path} ({size_mb:.1f} MB)")
print("Done.")
