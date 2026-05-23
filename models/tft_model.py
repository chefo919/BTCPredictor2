"""
Adaptive Temporal Fusion Transformer — macro direction model (TFT-ACB-XML).

Predicts 1-day price direction using 30 days of 4h-sampled context.
Features: h4/d1 indicators (24 features, all dynamic — no static covariates).
Input data comes pre-sampled at 4-hour boundaries from {YYYY}_tft_merged.csv,
so no internal downsampling is needed.

Reference: arXiv 2602.12380 — TFT-ACB-XML (Din & Khan, 2026)
"""
import os, sys
import numpy as np
import joblib

ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (SEQ_LEN_TFT as SEQ_LEN, HORIZON_TFT as HORIZON,
                    N_EPOCHS_TFT as N_EPOCHS, BATCH_TFT as BATCH,
                    STRIDE_TFT as STRIDE, D_MODEL, N_HEADS, DROPOUT_TFT as DROPOUT,
                    N_ENSEMBLE, TFT_SAMPLE_INTERVAL)

HORIZON_ROWS = HORIZON // TFT_SAMPLE_INTERVAL  # 1440 min / 240 min-per-row = 6 rows (1 day)

MODEL_DIR        = os.path.join(ROOT, "models", "saved")
MODEL_PATH       = os.path.join(MODEL_DIR, "tft.keras")
SCALER_PATH      = os.path.join(MODEL_DIR, "tft_scaler.pkl")
CHECKPOINT_DIR   = os.path.join(MODEL_DIR, "checkpoints_tft")
N_STATIC_PATH    = os.path.join(MODEL_DIR, "tft_n_static.txt")

def _seed_model_path(s):  return os.path.join(MODEL_DIR, f"tft_s{s}.keras")
def _seed_scaler_path(s): return os.path.join(MODEL_DIR, f"tft_scaler_s{s}.pkl")
def _seed_ckpt_dir(s):    return os.path.join(MODEL_DIR, f"checkpoints_tft_s{s}")
def _ensemble_ready():    return all(os.path.exists(_seed_model_path(s)) for s in range(N_ENSEMBLE))


import tensorflow as tf
from tensorflow.keras import layers, regularizers


class GRN(tf.keras.layers.Layer):
    """Gated Residual Network — core TFT building block."""
    def __init__(self, units, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        L2 = 1e-4
        self.dense1 = layers.Dense(units, activation="elu",
                                    kernel_regularizer=regularizers.l2(L2))
        self.dense2 = layers.Dense(units, kernel_regularizer=regularizers.l2(L2))
        self.gate   = layers.Dense(units, activation="sigmoid")
        self.norm   = layers.LayerNormalization()
        self.drop   = layers.Dropout(dropout)
        self.proj   = layers.Dense(units)

    def call(self, x, training=False):
        h = self.drop(self.dense1(x), training=training)
        h = self.dense2(h)
        g = self.gate(x)
        return self.norm(g * h + (1.0 - g) * self.proj(x))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.proj.units, "dropout": self.drop.rate})
        return cfg


class VSN(tf.keras.layers.Layer):
    """Variable Selection Network — learns which features matter per timestep."""
    def __init__(self, n_features, d_model, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_features   = n_features
        self.d_model      = d_model
        self.feature_grns = [GRN(d_model, dropout, name=f"grn_f{i}")
                              for i in range(n_features)]
        self.context_grn  = GRN(n_features, dropout, name="grn_ctx")
        self.softmax      = layers.Softmax(axis=-1)

    def call(self, x, training=False):
        weights = self.softmax(self.context_grn(x, training=training))
        # Accumulate weighted sum one feature at a time — avoids [B,T,n_feat,D] stack.
        out = self.feature_grns[0](x[..., 0:1], training=training) * weights[..., 0:1]
        for i in range(1, self.n_features):
            out = out + self.feature_grns[i](x[..., i:i+1], training=training) * weights[..., i:i+1]
        return out, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_features": self.n_features,
                    "d_model":    self.d_model,
                    "dropout":    self.feature_grns[0].drop.rate})
        return cfg


class _StaticGRN(tf.keras.layers.Layer):
    """Applies a GRN to the last timestep of static features, returning [B, D_MODEL]."""
    def __init__(self, n_static, d_model, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_static = n_static
        self.d_model  = d_model
        self.grn      = GRN(d_model, dropout, name="static_grn")

    def call(self, x, training=False):
        # x: [B, T, n_static] — take last timestep (values are ~constant over window)
        static_last = x[:, -1, :]           # [B, n_static]
        return self.grn(static_last, training=training)   # [B, D_MODEL]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_static": self.n_static, "d_model": self.d_model,
                    "dropout": self.grn.drop.rate})
        return cfg


def build_model(n_dynamic: int, n_static: int = 0) -> tf.keras.Model:
    """
    Single-input model: input shape = (SEQ_LEN, n_dynamic + n_static).
    When n_static=0 (no static covariates), skips the static GRN path entirely.
    """
    from tensorflow.keras import models

    n_total = n_dynamic + n_static
    inp = layers.Input(shape=(SEQ_LEN, n_total), name="seq_input")

    dyn_inp = inp[..., :n_dynamic]   # [B, T, n_dynamic]

    # ── Dynamic path: VSN -> LSTM -> Attention -> FFN ─────────────────────────
    vsn         = VSN(n_dynamic, D_MODEL, DROPOUT, name="vsn")
    x, _weights = vsn(dyn_inp)

    lstm_out = layers.LSTM(D_MODEL, return_sequences=True,
                            kernel_regularizer=regularizers.l2(1e-4),
                            name="lstm_enc")(x)
    x = layers.LayerNormalization(name="ln_lstm")(lstm_out + x)

    attn_out = layers.MultiHeadAttention(
        num_heads=N_HEADS, key_dim=D_MODEL // N_HEADS,
        dropout=DROPOUT, name="mhsa"
    )(x, x)
    attn_out = layers.Dropout(DROPOUT)(attn_out)
    x = layers.LayerNormalization(name="ln_attn")(attn_out + x)

    ffn = layers.Dense(D_MODEL * 4, activation="relu", name="ffn1")(x)
    ffn = layers.Dropout(DROPOUT)(ffn)
    ffn = layers.Dense(D_MODEL, name="ffn2")(ffn)
    x   = layers.LayerNormalization(name="ln_ffn")(ffn + x)

    pooled = layers.GlobalAveragePooling1D(name="pool")(x)   # [B, D_MODEL]

    # ── Static covariate path (skipped when n_static=0) ──────────────────────
    if n_static > 0:
        static_inp = inp[..., n_dynamic:]
        static_ctx = _StaticGRN(n_static, D_MODEL, DROPOUT, name="static_ctx")(static_inp)
        combined   = layers.Add(name="combine")([pooled, static_ctx])
    else:
        combined = pooled

    out = layers.Dense(1, activation="sigmoid",
                        kernel_regularizer=regularizers.l2(1e-3),
                        name="output")(combined)

    model = models.Model(inp, out, name="adaptive_tft")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


class _CheckpointCB(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir: str, prefix: str, every_n: int = 5):
        super().__init__()
        self._dir    = checkpoint_dir
        self._prefix = prefix
        self._every  = every_n

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self._every == 0:
            path = os.path.join(self._dir, f"{self._prefix}_epoch_{epoch+1:03d}.keras")
            self.model.save(path)


def _latest_checkpoint(ckpt_dir=None) -> tuple:
    ckpt_dir = ckpt_dir or CHECKPOINT_DIR
    if not os.path.exists(ckpt_dir):
        return None, 0
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith("tft_epoch_") and f.endswith(".keras")]
    if not files:
        return None, 0
    best, best_ep = None, 0
    for f in files:
        try:
            ep = int(f.split("_")[2].split(".")[0])
            if ep > best_ep:
                best_ep, best = ep, f
        except (IndexError, ValueError):
            continue
    return (os.path.join(ckpt_dir, best), best_ep) if best else (None, 0)


def _custom_objects():
    return {"GRN": GRN, "VSN": VSN, "_StaticGRN": _StaticGRN}


def train(feature_df, dynamic_cols: list, static_cols: list,
          seed: int = None) -> dict:
    import time
    from sklearn.preprocessing import StandardScaler

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    _model_path  = _seed_model_path(seed) if seed is not None else MODEL_PATH
    _scaler_path = _seed_scaler_path(seed) if seed is not None else SCALER_PATH
    _ckpt_dir    = _seed_ckpt_dir(seed)   if seed is not None else CHECKPOINT_DIR

    t0 = time.time()

    all_cols = dynamic_cols + static_cols
    df = feature_df.dropna(subset=all_cols).copy()
    df["target"] = (df["close"].shift(-HORIZON_ROWS) > df["close"]).astype(int)
    df = df.dropna(subset=["target"])

    # Input is already 4h-sampled from tft_merged — no downsampling needed here.
    print(f"  TFT training rows: {len(df):,} (4h-sampled)  label balance: {df['target'].mean():.3f}  "
          f"dynamic={len(dynamic_cols)} static={len(static_cols)}", flush=True)

    X = df[all_cols].values   # [n, n_dynamic + n_static]
    y = df["target"].values
    n = len(X)
    n_dynamic = len(dynamic_cols)
    n_static  = len(static_cols)

    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    # Recency augmentation
    recent_start = int(train_end * 0.75)
    X_tr_aug = np.concatenate([X[:train_end], X[recent_start:train_end]])
    y_tr_aug = np.concatenate([y[:train_end], y[recent_start:train_end]])
    perm     = np.random.permutation(len(X_tr_aug))
    X_tr_aug = X_tr_aug[perm]
    y_tr_aug = y_tr_aug[perm]
    print(f"  Recency augmentation: {train_end:,} → {len(X_tr_aug):,} rows", flush=True)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_tr_aug).astype(np.float32)
    X_val   = scaler.transform(X[train_end:val_end]).astype(np.float32)
    X_test  = scaler.transform(X[val_end:]).astype(np.float32)

    n_aug = len(X_train)

    def make_ds(X_sc, y_arr, start, end, shuffle=False):
        return tf.keras.utils.timeseries_dataset_from_array(
            X_sc,
            targets=y_arr[start + SEQ_LEN: end],
            sequence_length=SEQ_LEN,
            sequence_stride=STRIDE,
            batch_size=BATCH,
            shuffle=shuffle,
        ).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_train, y_tr_aug, 0, n_aug, shuffle=True)
    val_ds   = make_ds(X_val,   y, train_end, val_end)
    test_ds  = make_ds(X_test,  y, val_end, n)

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    os.makedirs(_ckpt_dir, exist_ok=True)
    ckpt_path, start_epoch = _latest_checkpoint(_ckpt_dir)
    expected_shape = (SEQ_LEN, n_dynamic + n_static)
    if ckpt_path:
        try:
            loaded = tf.keras.models.load_model(ckpt_path, custom_objects=_custom_objects())
            if tuple(loaded.input_shape[1:]) == expected_shape:
                model = loaded
                lbl = f"seed {seed}" if seed is not None else "TFT"
                print(f"  Resuming {lbl} from checkpoint epoch {start_epoch}", flush=True)
            else:
                print(f"  Checkpoint shape mismatch — starting fresh.", flush=True)
                model = build_model(n_dynamic, n_static)
                start_epoch = 0
        except Exception as e:
            print(f"  Checkpoint load failed ({e}) — starting fresh.", flush=True)
            model = build_model(n_dynamic, n_static)
            start_epoch = 0
    else:
        model = build_model(n_dynamic, n_static)
        start_epoch = 0

    print(f"  Parameters: {model.count_params():,}", flush=True)

    BAR = 36

    class _ProgressCB(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs    = logs or {}
            done    = epoch + 1 + start_epoch
            elapsed = time.time() - self._t0
            rate    = elapsed / (epoch + 1)
            eta     = rate * (N_EPOCHS - done)
            filled  = int(BAR * done / N_EPOCHS)
            bar     = "#" * filled + "-" * (BAR - filled)
            print(
                f"  Epoch {done:3d}/{N_EPOCHS} [{bar}] | "
                f"Loss: {logs.get('loss',0):.4f} | "
                f"Val Loss: {logs.get('val_loss',0):.4f} | "
                f"Acc: {logs.get('accuracy',0):.3f} | "
                f"Val Acc: {logs.get('val_accuracy',0):.3f} | "
                f"Elapsed: {_fmt(elapsed)} | ETA: {_fmt(eta)}",
                flush=True,
            )

    callbacks = [
        _ProgressCB(),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.3, min_lr=1e-6, verbose=0),
        _CheckpointCB(_ckpt_dir, "tft", every_n=5),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=N_EPOCHS, initial_epoch=start_epoch,
                        callbacks=callbacks, verbose=0)

    val_acc  = max(history.history.get("val_accuracy", [0.5]))
    _, test_acc = model.evaluate(test_ds, verbose=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(_model_path)
    joblib.dump(scaler, _scaler_path)
    with open(N_STATIC_PATH, "w") as f:
        f.write(str(n_static))
    # Also save to primary path when no seed (backward compat)
    if seed is None:
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    return {"test_acc": float(test_acc), "val_acc": float(val_acc),
            "time_taken": time.time() - t0}


def train_ensemble(feature_df, dynamic_cols: list, static_cols: list) -> list:
    """Train N_ENSEMBLE seeds and return list of results. Predictions are averaged at inference."""
    results = []
    for s in range(N_ENSEMBLE):
        print(f"\n{'='*50}", flush=True)
        print(f"TFT ENSEMBLE — seed {s+1}/{N_ENSEMBLE}", flush=True)
        print(f"{'='*50}", flush=True)
        r = train(feature_df, dynamic_cols, static_cols, seed=s)
        results.append(r)
        print(f"  Seed {s}: val={r['val_acc']:.4f}  test={r['test_acc']:.4f}", flush=True)
    avg_val  = float(np.mean([r["val_acc"]  for r in results]))
    avg_test = float(np.mean([r["test_acc"] for r in results]))
    print(f"\n  Ensemble average: val={avg_val:.4f}  test={avg_test:.4f}", flush=True)
    return results


# ── Inference ─────────────────────────────────────────────────────────────────

def _load_n_static() -> int:
    if os.path.exists(N_STATIC_PATH):
        with open(N_STATIC_PATH) as f:
            return int(f.read().strip())
    return 0    # default: no static covariates


def _infer_batch(model, scaler, X_dynamic, X_static, batch_size):
    """
    Run batch inference for one model. Input is already 4h-sampled from tft_merged.
    No downsampling needed.
    """
    X_all = np.concatenate([X_dynamic, X_static], axis=1)
    X_sc  = scaler.transform(X_all).astype(np.float32)
    n     = len(X_sc)
    probs = np.full(n, 0.5, dtype=np.float32)
    n_batches = max(1, (n - SEQ_LEN + batch_size - 1) // batch_size)
    for b in range(n_batches):
        b_start = b * batch_size
        b_end   = min(b_start + batch_size, n - SEQ_LEN)
        if b_start >= b_end:
            break
        abs_idx = np.arange(SEQ_LEN + b_start, SEQ_LEN + b_end)
        win_idx = abs_idx[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
        preds   = model.predict(X_sc[win_idx], verbose=0, batch_size=batch_size)
        probs[SEQ_LEN + b_start: SEQ_LEN + b_end] = preds.flatten()
    return probs


def predict_proba_batch(X_dynamic: np.ndarray, X_static: np.ndarray,
                         batch_size: int = 512) -> np.ndarray:
    """
    Batch inference. Auto-uses ensemble (N_ENSEMBLE seeds averaged) if trained,
    otherwise falls back to single model. Input must be pre-sampled at 4h intervals.
    """
    if _ensemble_ready():
        all_probs = []
        for s in range(N_ENSEMBLE):
            m = tf.keras.models.load_model(_seed_model_path(s), custom_objects=_custom_objects())
            sc = joblib.load(_seed_scaler_path(s))
            all_probs.append(_infer_batch(m, sc, X_dynamic, X_static, batch_size))
            del m
        return np.mean(all_probs, axis=0)

    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)
    return _infer_batch(model, scaler, X_dynamic, X_static, batch_size)


def predict_proba(X_dynamic: np.ndarray, X_static: np.ndarray) -> float:
    """
    Single-sample inference. Auto-uses ensemble if trained.
    X_dynamic must have at least SEQ_LEN rows (already 4h-sampled).
    """
    X_all = np.concatenate([X_dynamic, X_static], axis=1)
    X_h   = X_all[-SEQ_LEN:]

    if _ensemble_ready():
        preds = []
        for s in range(N_ENSEMBLE):
            m  = tf.keras.models.load_model(_seed_model_path(s), custom_objects=_custom_objects())
            sc = joblib.load(_seed_scaler_path(s))
            inp = sc.transform(X_h).astype(np.float32).reshape(1, SEQ_LEN, -1)
            preds.append(float(m.predict(inp, verbose=0)[0][0]))
            del m
        return float(np.mean(preds))

    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)
    inp    = scaler.transform(X_h).astype(np.float32).reshape(1, SEQ_LEN, -1)
    return float(model.predict(inp, verbose=0)[0][0])


def is_trained() -> bool:
    return _ensemble_ready() or (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH))
