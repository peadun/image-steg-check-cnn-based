"""
EfficientNetB0 Training Script — ALASKA#2 JPEG Steganalysis
TensorFlow 2.x

Implements the architecture from:
  Hong et al., "Lightweight image steganalysis with block-wise pruning",
  Scientific Reports 2023.  https://doi.org/10.1038/s41598-023-43386-2

Key modifications over vanilla EfficientNetB0:
  1. Stem stride ablation  — first Conv2D stride 2→1 (preserves stego signal)
  2. Block-wise pruning    — remove last REMOVE_N MBConv blocks (rm-8 optimal)
  3. End-to-end fine-tuning from ImageNet weights
  4. AdamW lr=1e-4, no LR scheduler, no weight decay  (per paper)

Dataset: ALASKA#2  (Cover/ + JMiPOD/ + JUNIWARD/ + UERD/)
  — 75k covers × 3 algorithms = 225k cover-stego pairs
  — Images are already 512×512 JPEG, no preprocessing needed
  — Split 80/10/10 (train/valid/test) from the 75k covers

Input:  RGB JPEG [0,255] float32 — EfficientNetB0 rescales internally.
Output: Dense(2) logits → class 0 = cover, class 1 = stego.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import time
import random
from random import shuffle, random as rand

import numpy as np
from glob import glob
from PIL import Image

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(42)
np.random.seed(42)

# ==============================================================================
# CONFIGURATION — update the two paths marked UPDATE before running
# ==============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
# UPDATE: root folder that contains Cover/, JMiPOD/, JUNIWARD/, UERD/
ALASKA2_ROOT = '/workspace/data/ALASKA2'

COVER_DIR  = os.path.join(ALASKA2_ROOT, 'Cover')
STEGO_DIRS = [
    os.path.join(ALASKA2_ROOT, 'JMiPOD'),
    os.path.join(ALASKA2_ROOT, 'JUNIWARD'),
    os.path.join(ALASKA2_ROOT, 'UERD'),
]

LOG_DIR = '/workspace/logs/EfficientNet_ALASKA2_8thrun'

# ── Dataset split (applied to the 75k covers before building pairs) ───────────
SPLIT_SEED = 42
TRAIN_FRAC = 0.80   # ~60k covers → ~180k pairs
VALID_FRAC = 0.10   # ~7.5k covers → ~22.5k pairs
# TEST_FRAC  = 0.10  # ~7.5k covers → ~22.5k pairs  (remainder)

# ── Model ─────────────────────────────────────────────────────────────────────
REMOVE_N = 7    # FIX: TF Keras has 16 MBConv blocks; remove 7 → keep 9 blocks = paper's rm-8 (418k params)

# ── Input size ────────────────────────────────────────────────────────────────
# ALASKA2 images are 512×512. Set CROP_SIZE=512 on the cloud GPU (4090).
# For local pipeline testing on a 1070 (8 GB) set CROP_SIZE=256 and
# TRAIN_BATCH_SIZE=8 — just to verify the script runs before renting.
CROP_SIZE = 512

# ── Batch sizes ───────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE  = 16

# ── Training length ───────────────────────────────────────────────────────────
# Paper trains 50 epochs on ALASKA2 with batch=48.
# ~60k covers × 3 algos = 180k pairs → 360k samples/epoch
# Steps/epoch ≈ 360k / 16 ≈ 22,500  →  50 epochs ≈ 1,125,000 steps
MAX_ITER = 1125000

# ── LR: AdamW, constant with manual step-based decay ─────────────────────────
# Keep LR_VALUES as a single-element list so build_lr_schedule returns a float
# — this preserves the optimizer's internal variable layout that the existing
# checkpoint was saved with. Swapping to PiecewiseConstantDecay changes the
# optimizer's _variables list and corrupts state on restore (→ NaN loss).
# Decay is applied manually via LR_DECAY_SCHEDULE inside the training loop.
LR_BOUNDARIES = []
LR_VALUES     = [5e-5]

# Step-based LR decay applied in the train loop via optimizer.learning_rate.assign().
# Run 8 plateaued at ~74.2% val (peak 74.88%) at step 532.5k with train acc
# still 3 pp below val (no overfitting → pure optimisation plateau).
# First decay at 600k (5e-5 → 1e-5) damped the noise (±1.5 pp → ±0.2 pp) and
# lifted peak to 76.30% at step 690k, but by step 712.5k the model is locked in
# a tight 76.0-76.3 band for 14+ consecutive checkpoints. 1e-5 phase has done
# its job; moving the second decay up from 900k → 720k so we don't burn budget
# at a known-flat LR.
LR_DECAY_SCHEDULE = [
    (600000, 1e-5),
    (720000, 2e-6),
]

# ── Logging intervals ─────────────────────────────────────────────────────────
TRAIN_INTERVAL = 200
VALID_INTERVAL = 7500    # once per epoch
SAVE_INTERVAL  = 7500    # once per epoch

# Set to a checkpoint path to resume from a specific checkpoint,
# or leave as None to auto-resume from the latest saved checkpoint.
LOAD_PATH = None


# ==============================================================================
# DATASET UTILITIES
# ==============================================================================

def split_covers(cover_dir, train_frac=TRAIN_FRAC, valid_frac=VALID_FRAC,
                 seed=SPLIT_SEED):
    """
    Load all cover image paths, shuffle deterministically, and split into
    train / valid / test lists.  The same seed always produces the same split
    so checkpointing works correctly across runs.

    Returns
    -------
    train_covers, valid_covers, test_covers : lists of absolute file paths
    """
    covers = sorted(glob(os.path.join(cover_dir, '*.jpg')))
    if not covers:
        covers = sorted(glob(os.path.join(cover_dir, '*.JPEG')))
    assert covers, f'No JPEG images found in {cover_dir}'

    rng = random.Random(seed)
    rng.shuffle(covers)

    n       = len(covers)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    train = covers[:n_train]
    valid = covers[n_train : n_train + n_valid]
    test  = covers[n_train + n_valid :]
    return train, valid, test


def scan_stego_dirs(stego_dirs):
    """
    Scan each stego directory once and return a list of (dir, filename_set)
    tuples.  Call this once and pass the result to build_pairs() so the
    directories are not re-scanned for every split (train / valid / test).
    """
    print('  Scanning stego directories …', flush=True)
    stego_sets = []
    for stego_dir in stego_dirs:
        filenames = {os.path.basename(p)
                     for p in glob(os.path.join(stego_dir, '*.jpg'))}
        print(f'    {len(filenames):,} files  ←  {stego_dir}')
        stego_sets.append((stego_dir, filenames))
    return stego_sets


def build_pairs(cover_list, stego_sets):
    """
    For each cover image path and each stego directory, create a
    (cover_path, stego_path) pair — provided the stego file exists.

    ALASKA2 uses identical filenames across all directories, e.g.:
      Cover/00001.jpg  ↔  JMiPOD/00001.jpg
                        ↔  JUNIWARD/00001.jpg
                        ↔  UERD/00001.jpg

    Parameters
    ----------
    stego_sets : output of scan_stego_dirs() — list of (dir, filename_set)

    Returns a flat list of (cover_path, stego_path) tuples.
    Each cover appears once per algorithm → 1:1 balance within each pair.
    """
    pairs = []
    missing = 0
    for cover_path in cover_list:
        fname = os.path.basename(cover_path)
        for stego_dir, filenames in stego_sets:
            if fname in filenames:
                pairs.append((cover_path, os.path.join(stego_dir, fname)))
            else:
                missing += 1
    if missing:
        print(f'  WARNING: {missing} stego files not found and skipped.')
    return pairs


# ==============================================================================
# DATA GENERATORS
# ==============================================================================

def _load_pair(cover_path, stego_path):
    """Load a cover/stego pair as (2, H, W, 3) uint8 numpy array."""
    c = np.array(Image.open(cover_path).convert('RGB'), dtype='uint8')
    s = np.array(Image.open(stego_path).convert('RGB'), dtype='uint8')
    return np.stack([c, s], axis=0)   # (2, H, W, 3)


def _crop_pair(batch, crop=CROP_SIZE, center=False):
    """
    Crop both images in a (2, H, W, 3) batch to crop×crop.
    For ALASKA2 at 512×512 with CROP_SIZE=512 this is a no-op.
    For local testing with CROP_SIZE=256 it applies a random/centre crop.
    """
    H, W = batch.shape[1], batch.shape[2]
    if crop >= H and crop >= W:
        return batch
    if center:
        y = (H - crop) // 2
        x = (W - crop) // 2
    else:
        y = random.randint(0, H - crop)
        x = random.randint(0, W - crop)
    return batch[:, y:y + crop, x:x + crop, :]


def gen_train(pairs):
    """
    Infinite iterator of (image, label) samples for training.
    Yields two samples per pair: (cover, 0) then (stego, 1).
    Applies random crop (if CROP_SIZE < 512), horizontal flip,
    vertical flip, and 90° rotation — same transform to both images.
    """
    pairs = list(pairs)
    while True:
        shuffle(pairs)
        for cover_path, stego_path in pairs:
            batch = _load_pair(cover_path, stego_path)
            batch = _crop_pair(batch, center=False)

            # Augmentation — applied identically to both images
            if rand() > 0.5:
                batch = np.flip(batch, axis=1)   # vertical flip
            if rand() > 0.5:
                batch = np.flip(batch, axis=2)   # horizontal flip
            k = random.randint(0, 3)
            batch = np.rot90(batch, k=k, axes=(1, 2))

            yield batch[0].copy().astype(np.float32), np.int32(0)  # cover
            yield batch[1].copy().astype(np.float32), np.int32(1)  # stego


def gen_eval(pairs):
    """
    Infinite iterator of (image, label) samples for validation/test.
    No augmentation; centre crop only (no-op at 512×512).
    """
    pairs = list(pairs)
    while True:
        for cover_path, stego_path in pairs:
            batch = _load_pair(cover_path, stego_path)
            batch = _crop_pair(batch, center=True)
            yield batch[0].copy().astype(np.float32), np.int32(0)
            yield batch[1].copy().astype(np.float32), np.int32(1)


def make_dataset(gen_fn, pairs, batch_size, shuffle_buffer=0):
    """Wrap a pair-generator in a tf.data pipeline."""
    ds = tf.data.Dataset.from_generator(
        lambda: gen_fn(pairs),
        output_signature=(
            tf.TensorSpec(shape=(CROP_SIZE, CROP_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ==============================================================================
# MODEL
# ==============================================================================

def build_efficientnet_model(input_shape=(CROP_SIZE, CROP_SIZE, 3),
                             num_classes=2,
                             remove_n=REMOVE_N):
    """
    EfficientNetB0 for steganalysis — Hong et al., Sci. Reports 2023.

    Modifications applied to vanilla EfficientNetB0:
      1. Stem stride ablation : first Conv2D stride 2→1 so the 4× larger
         feature map from the stem captures the subtle stego noise before
         downsampling operations suppress it.
      2. Block-wise pruning   : keep only the first (17 − remove_n) MBConv
         blocks.  rm-8 (9 blocks) is optimal — ~10× smaller, same accuracy.

    Training: full fine-tuning from ImageNet weights, no freeze phase.
    Input : float32 [0, 255] — internal rescaling/normalisation applied.
    Output: logits (batch, 2)
    """
    # ── load pretrained base ─────────────────────────────────────────────────
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=input_shape
    )

    # ── find block endpoint names for truncation ─────────────────────────────
    # Each MBConv block ends with either '_add' (residual path) or
    # '_project_bn' (no residual).  Some blocks have BOTH — iterate layers
    # in graph order and keep the LAST endpoint seen per block ID so residual
    # blocks are counted exactly once (at their '_add', not also '_project_bn').
    from collections import OrderedDict
    block_map = OrderedDict()
    for l in base.layers:
        name = l.name
        if name.startswith('block') and (name.endswith('_add') or
                                          name.endswith('_project_bn')):
            block_id = name.split('_')[0]        # e.g. 'block2b'
            block_map[block_id] = name           # overwrite → keeps last seen
    block_ends = list(block_map.values())        # one entry per MBConv block

    n_keep = max(1, len(block_ends) - remove_n)
    cutoff = block_ends[n_keep - 1]
    print(f'  EfficientNetB0: {len(block_ends)} MBConv blocks '
          f'→ keeping first {n_keep}  (cutoff: {cutoff})')

    # ── stem stride ablation via config round-trip ───────────────────────────
    try:
        cfg = base.get_config()
        for lc in cfg['layers']:
            if lc.get('name') == 'stem_conv':
                lc['config']['strides'] = [1, 1]
                break
        ablated = tf.keras.Model.from_config(cfg)
        for layer in ablated.layers:
            try:
                src_w = base.get_layer(layer.name).get_weights()
                if src_w:
                    layer.set_weights(src_w)
            except (ValueError, AttributeError):
                pass
        feature_src = ablated
        print('  Stem stride ablation: 2 → 1  ✓')
    except Exception as exc:
        print(f'  Stem stride ablation: skipped ({exc})')
        feature_src = base

    feature_src.trainable = True

    # ── truncated feature extractor ──────────────────────────────────────────
    trunk = tf.keras.Model(
        inputs=feature_src.input,
        outputs=feature_src.get_layer(cutoff).output,
        name='efficientnet_trunk'
    )

    # ── steganalysis head ────────────────────────────────────────────────────
    inputs  = tf.keras.Input(shape=input_shape, dtype=tf.float32, name='input_image')
    x       = trunk(inputs)
    x       = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    outputs = tf.keras.layers.Dense(num_classes, name='logits')(x)
    # Note: dropout removed — paper does not use it, and with 225k pairs
    # the model has sufficient data without it.

    return tf.keras.Model(inputs, outputs, name='efficientnet_steg')


# ==============================================================================
# HELPERS
# ==============================================================================

def get_lr(opt):
    lr = opt.learning_rate
    if callable(lr):
        return float(lr(opt.iterations).numpy())
    return float(tf.keras.backend.get_value(lr))


def build_lr_schedule(boundaries, values):
    if not boundaries:
        return values[0]
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=values
    )


def make_train_step(model, optimizer, loss_fn):
    @tf.function
    def train_step(imgs, labels):
        with tf.GradientTape() as tape:
            logits = model(imgs, training=True)
            loss   = loss_fn(labels, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        preds    = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return loss, accuracy
    return train_step


# ==============================================================================
# TRAINING
# ==============================================================================

def train():
    print('\n' + '=' * 70)
    print('EfficientNetB0  –  ALASKA#2 JPEG Steganalysis  –  TensorFlow 2.x')
    print('=' * 70)

    # ── verify directories ───────────────────────────────────────────────────
    print('\nChecking directories …')
    all_dirs = [COVER_DIR] + STEGO_DIRS
    for d in all_dirs:
        if not os.path.isdir(d):
            print(f'  ✗ NOT FOUND: {d}')
            return
        n = len(glob(os.path.join(d, '*.jpg')))
        print(f'  ✓ {n:6d} images  →  {d}')

    # ── split covers → build pairs ───────────────────────────────────────────
    print('\nSplitting covers and building pairs …')
    train_covers, valid_covers, test_covers = split_covers(COVER_DIR)

    stego_sets  = scan_stego_dirs(STEGO_DIRS)   # scan once, reuse for all splits
    train_pairs = build_pairs(train_covers, stego_sets)
    valid_pairs = build_pairs(valid_covers, stego_sets)

    n_train_pairs = len(train_pairs)   # ~180k
    n_valid_pairs = len(valid_pairs)   # ~22.5k

    print(f'\n  Train covers : {len(train_covers):,}  →  {n_train_pairs:,} pairs')
    print(f'  Valid covers : {len(valid_covers):,}  →  {n_valid_pairs:,} pairs')
    print(f'  Test  covers : {len(test_covers):,}  →  (use --mode test to evaluate)')
    print(f'\n  Samples/epoch  : {n_train_pairs * 2:,}')
    print(f'  Steps/epoch    : {(n_train_pairs * 2) // TRAIN_BATCH_SIZE:,}')
    print(f'  Crop size      : {CROP_SIZE}×{CROP_SIZE}')
    print(f'  Batch size     : {TRAIN_BATCH_SIZE}')
    print(f'  rm-{REMOVE_N}  |  MAX_ITER: {MAX_ITER:,}')

    os.makedirs(LOG_DIR, exist_ok=True)

    # ── datasets ─────────────────────────────────────────────────────────────
    print('\nBuilding data pipelines …')
    train_ds = make_dataset(gen_train, train_pairs, TRAIN_BATCH_SIZE, shuffle_buffer=512)
    valid_ds = make_dataset(gen_eval,  valid_pairs, VALID_BATCH_SIZE)

    # Sanity check on one batch
    imgs0, labels0 = next(iter(train_ds))
    l0 = labels0.numpy()
    print(f'\nSanity check:')
    print(f'  batch shape : {imgs0.shape}  dtype : {imgs0.dtype}')
    print(f'  pixel range : [{imgs0.numpy().min():.0f}, {imgs0.numpy().max():.0f}]  '
          f'(expect [0, 255])')
    print(f'  label balance : cover={( l0 == 0).sum()}  stego={(l0 == 1).sum()}')

    # ── model ────────────────────────────────────────────────────────────────
    print(f'\nBuilding EfficientNetB0 (rm-{REMOVE_N}, stem-ablated) …')
    model = build_efficientnet_model()
    model.summary(line_length=80)

    # ── optimiser / loss ─────────────────────────────────────────────────────
    lr_schedule = build_lr_schedule(LR_BOUNDARIES, LR_VALUES)
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=0.0
        )
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, epsilon=1e-8
        )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # ── checkpoint ───────────────────────────────────────────────────────────
    ckpt     = tf.train.Checkpoint(optimizer=optimizer, model=model,
                                   step=tf.Variable(0, dtype=tf.int64))
    ckpt_mgr = tf.train.CheckpointManager(ckpt, LOG_DIR, max_to_keep=None)

    if LOAD_PATH:
        ckpt.restore(LOAD_PATH).expect_partial()
        print(f'Restored checkpoint : {LOAD_PATH}')
    elif ckpt_mgr.latest_checkpoint:
        ckpt.restore(ckpt_mgr.latest_checkpoint).expect_partial()
        print(f'Auto-restored : {ckpt_mgr.latest_checkpoint}')
    else:
        print('Training from scratch.')

    start_step = int(ckpt.step.numpy())

    # ── TensorBoard ──────────────────────────────────────────────────────────
    tb_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, 'tb'))

    # ── compiled steps ───────────────────────────────────────────────────────
    train_step = make_train_step(model, optimizer, loss_fn)

    @tf.function
    def valid_step(imgs, labels):
        logits   = model(imgs, training=False)
        loss     = loss_fn(labels, logits)
        preds    = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        return loss, accuracy

    def run_validation():
        vloss = tf.keras.metrics.Mean()
        vacc  = tf.keras.metrics.Mean()
        n_steps = (n_valid_pairs * 2) // VALID_BATCH_SIZE
        for imgs, labels in valid_ds.take(n_steps):
            l, a = valid_step(imgs, labels)
            vloss.update_state(l)
            vacc.update_state(a)
        return vloss.result().numpy(), vacc.result().numpy()

    # ── accumulators ─────────────────────────────────────────────────────────
    train_loss_acc = tf.keras.metrics.Mean()
    train_acc_acc  = tf.keras.metrics.Mean()

    # ── main loop ─────────────────────────────────────────────────────────────
    print(f'\nStarting training from step {start_step + 1} → {MAX_ITER}')
    print(f'  Validation  every {VALID_INTERVAL} steps  (~every epoch)')
    print(f'  Checkpoints every {SAVE_INTERVAL} steps  →  {LOG_DIR}\n')

    # Apply any decay boundaries already crossed (when resuming past them).
    for boundary, new_lr in LR_DECAY_SCHEDULE:
        if start_step >= boundary:
            optimizer.learning_rate.assign(new_lr)
    print(f'  Initial LR at resume: {get_lr(optimizer):.2e}\n')

    train_iter = iter(train_ds)
    t0 = time.time()

    for step in range(start_step + 1, MAX_ITER + 1):

        # Manual LR decay at boundary crossings. We do not use
        # PiecewiseConstantDecay as the scheduler because switching the
        # optimizer's learning_rate from a Variable to a schedule changes
        # the optimizer's _variables layout, which breaks checkpoint restore.
        for boundary, new_lr in LR_DECAY_SCHEDULE:
            if step == boundary:
                optimizer.learning_rate.assign(new_lr)
                print(f'\n  [LR decay] step={step}: LR → {new_lr:.2e}\n')

        imgs, labels = next(train_iter)
        loss, acc    = train_step(imgs, labels)
        ckpt.step.assign_add(1)

        train_loss_acc.update_state(loss)
        train_acc_acc.update_state(acc)

        # ── print training metrics ────────────────────────────────────────────
        if step % TRAIN_INTERVAL == 0:
            elapsed  = time.time() - t0
            it_sec   = TRAIN_INTERVAL / elapsed
            avg_loss = train_loss_acc.result().numpy()
            avg_acc  = train_acc_acc.result().numpy()
            lr_now   = get_lr(optimizer)

            print(f'Step {step:>7d}/{MAX_ITER}  '
                  f'loss={avg_loss:.4f}  acc={avg_acc * 100:.2f}%  '
                  f'lr={lr_now:.2e}  speed={it_sec:.1f} it/s')

            with tb_writer.as_default():
                tf.summary.scalar('train/loss', avg_loss, step=step)
                tf.summary.scalar('train/acc',  avg_acc,  step=step)
                tf.summary.scalar('train/lr',   lr_now,   step=step)

            train_loss_acc.reset_state()
            train_acc_acc.reset_state()
            t0 = time.time()

        # ── validation ────────────────────────────────────────────────────────
        if step % VALID_INTERVAL == 0:
            vl, va = run_validation()
            pe     = 1.0 - va
            print(f'\n  [Valid @ step {step}]  '
                  f'loss={vl:.4f}  acc={va * 100:.2f}%  PE={pe:.4f}\n')
            with tb_writer.as_default():
                tf.summary.scalar('valid/loss', vl, step=step)
                tf.summary.scalar('valid/acc',  va, step=step)
                tf.summary.scalar('valid/pe',   pe, step=step)

        # ── checkpoint ────────────────────────────────────────────────────────
        if step % SAVE_INTERVAL == 0:
            ckpt.step.assign(step)
            saved = ckpt_mgr.save(checkpoint_number=step)
            print(f'  [Saved]  {saved}')

    print('\nTraining complete!')
    vl, va = run_validation()
    print(f'Final valid  →  accuracy={va * 100:.2f}%   PE={1.0 - va:.4f}')


# ==============================================================================
# TESTING
# ==============================================================================

def test(checkpoint_path=None):
    print('\n' + '=' * 70)
    print('EfficientNetB0  –  ALASKA#2  –  Testing')
    print('=' * 70)

    # Rebuild the same deterministic split to get the test covers
    print('\nRebuilding split to recover test set …')
    _, _, test_covers = split_covers(COVER_DIR)
    test_pairs = build_pairs(test_covers, scan_stego_dirs(STEGO_DIRS))
    n_test_pairs = len(test_pairs)
    print(f'  Test covers : {len(test_covers):,}  →  {n_test_pairs:,} pairs')

    if (n_test_pairs * 2) % TEST_BATCH_SIZE != 0:
        # Trim to nearest multiple so drop_remainder doesn't skip samples
        keep = ((n_test_pairs * 2) // TEST_BATCH_SIZE) * TEST_BATCH_SIZE
        trim_pairs = n_test_pairs - keep // 2
        test_pairs = test_pairs[:len(test_pairs) - trim_pairs]
        n_test_pairs = len(test_pairs)
        print(f'  Trimmed to {n_test_pairs:,} pairs for clean batch division.')

    test_ds = make_dataset(gen_eval, test_pairs, TEST_BATCH_SIZE)

    model    = build_efficientnet_model()
    ckpt     = tf.train.Checkpoint(model=model)
    ckpt_mgr = tf.train.CheckpointManager(ckpt, LOG_DIR, max_to_keep=None)

    load = checkpoint_path or ckpt_mgr.latest_checkpoint
    if not load:
        print(f'ERROR: no checkpoint found in {LOG_DIR}')
        return

    ckpt.restore(load).expect_partial()
    print(f'Loaded checkpoint: {load}')

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    tloss   = tf.keras.metrics.Mean()
    tacc    = tf.keras.metrics.Mean()
    n_steps = (n_test_pairs * 2) // TEST_BATCH_SIZE

    for imgs, labels in test_ds.take(n_steps):
        logits = model(imgs, training=False)
        loss   = loss_fn(labels, logits)
        preds  = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc    = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
        tloss.update_state(loss)
        tacc.update_state(acc)

    accuracy = tacc.result().numpy()
    pe       = 1.0 - accuracy
    print(f'\nTest Results:')
    print(f'  Accuracy : {accuracy * 100:.2f}%')
    print(f'  Loss     : {tloss.result().numpy():.4f}')
    print(f'  PE       : {pe:.4f}  ({pe * 100:.2f}%)')


def _d4_orientations_batch(imgs):
    """
    Return a list of 8 D4 dihedral orientations of a batch of images.
    imgs shape: (B, H, W, 3) — output: list of 8 tensors each (B, H, W, 3).
    """
    out = []
    for flip in (False, True):
        x = tf.reverse(imgs, axis=[2]) if flip else imgs       # horizontal flip
        for k in range(4):
            out.append(tf.image.rot90(x, k=k))                 # 0/90/180/270°
    return out


def test_ensemble(checkpoint_paths, use_tta=False, n_pairs_limit=None):
    """
    Ensemble evaluation: load N checkpoints into N model instances, run each
    on the test set (optionally with 8-orientation TTA per model), average the
    softmax outputs, compute accuracy.

    Parameters
    ----------
    checkpoint_paths : list[str]
        Paths to checkpoints to ensemble (e.g. ckpt-585000, ckpt-690000, ...).
    use_tta : bool
        If True, run each model on all 8 D4 dihedral orientations and average
        across both orientations and models. Multiplies inference cost ×8.
    n_pairs_limit : int or None
        If set, only evaluate on this many test pairs (for quick experiments).
    """
    print('\n' + '=' * 70)
    print(f'EfficientNetB0 – ALASKA#2 – Ensemble Test '
          f'({len(checkpoint_paths)} models, TTA={use_tta})')
    print('=' * 70)

    print('\nRebuilding split to recover test set …')
    _, _, test_covers = split_covers(COVER_DIR)
    test_pairs = build_pairs(test_covers, scan_stego_dirs(STEGO_DIRS))

    if n_pairs_limit is not None and len(test_pairs) > n_pairs_limit:
        test_pairs = test_pairs[:n_pairs_limit]
        print(f'  Limited to first {n_pairs_limit:,} pairs for this run.')

    n_test_pairs = len(test_pairs)
    if (n_test_pairs * 2) % TEST_BATCH_SIZE != 0:
        keep = ((n_test_pairs * 2) // TEST_BATCH_SIZE) * TEST_BATCH_SIZE
        trim_pairs = n_test_pairs - keep // 2
        test_pairs = test_pairs[:len(test_pairs) - trim_pairs]
        n_test_pairs = len(test_pairs)
    print(f'  Test pairs : {n_test_pairs:,}  ({n_test_pairs * 2:,} samples)')

    test_ds = make_dataset(gen_eval, test_pairs, TEST_BATCH_SIZE)

    # Build N model instances and load each checkpoint.
    models = []
    for path in checkpoint_paths:
        m = build_efficientnet_model()
        ckpt = tf.train.Checkpoint(model=m)
        ckpt.restore(path).expect_partial()
        models.append(m)
        print(f'  Loaded: {path}')

    correct = 0
    total   = 0
    n_steps = (n_test_pairs * 2) // TEST_BATCH_SIZE
    log_every = max(1, n_steps // 20)

    print(f'\nRunning {n_steps:,} batches × {len(models)} models'
          f'{" × 8 orientations" if use_tta else ""} …')
    t0 = time.time()

    for step, (imgs, labels) in enumerate(test_ds.take(n_steps)):
        # Generate orientations once (or single batch if no TTA).
        batches = _d4_orientations_batch(imgs) if use_tta else [imgs]

        # Sum softmax probs across all (model × orientation) combinations.
        prob_sum = tf.zeros((TEST_BATCH_SIZE, 2), dtype=tf.float32)
        n_passes = 0
        for m in models:
            for b in batches:
                logits = m(b, training=False)
                prob_sum += tf.nn.softmax(logits, axis=-1)
                n_passes += 1
        avg_probs = prob_sum / float(n_passes)

        preds   = tf.argmax(avg_probs, axis=1, output_type=tf.int32)
        correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32)))
        total   += TEST_BATCH_SIZE

        if (step + 1) % log_every == 0 or step + 1 == n_steps:
            running_acc = correct / total
            elapsed     = time.time() - t0
            print(f'  step {step + 1:>5d}/{n_steps}  '
                  f'running_acc={running_acc * 100:.2f}%  '
                  f'elapsed={elapsed / 60:.1f} min')

    accuracy = correct / total
    pe       = 1.0 - accuracy
    print(f'\nEnsemble Test Results:')
    print(f'  Models   : {len(models)} ({"with" if use_tta else "without"} TTA)')
    print(f'  Pairs    : {n_test_pairs:,}')
    print(f'  Accuracy : {accuracy * 100:.2f}%')
    print(f'  PE       : {pe:.4f}  ({pe * 100:.2f}%)')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EfficientNetB0 – ALASKA#2 JPEG Steganalysis'
    )
    parser.add_argument('--mode', choices=['train', 'test', 'test_ensemble'],
                        default='train')
    parser.add_argument('--checkpoint', default=None,
                        help='Specific checkpoint path for --mode test')
    parser.add_argument('--checkpoints', nargs='+', default=None,
                        help='Space-separated list of checkpoint paths for '
                             '--mode test_ensemble')
    parser.add_argument('--tta', action='store_true',
                        help='Enable 8-orientation D4 TTA in test_ensemble')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional cap on number of test pairs')
    args = parser.parse_args()

    tf.get_logger().setLevel('ERROR')

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test(args.checkpoint)
    else:  # test_ensemble
        if not args.checkpoints:
            raise SystemExit('ERROR: --mode test_ensemble requires '
                             '--checkpoints <path1> <path2> …')
        test_ensemble(args.checkpoints, use_tta=args.tta,
                      n_pairs_limit=args.limit)
