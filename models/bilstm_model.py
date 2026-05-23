"""
Attention-Customized Bidirectional LSTM (ACB) — short-term momentum model.

Inputs:  15m, 1H features (24 cols) — momentum signals sampled at 15-minute intervals
Context: 24 hours at 15min resolution (SEQ_LEN=96 rows from bilstm_merged)
Target:  1-day price direction (HORIZON=1440)

Input data comes pre-sampled at 15-minute boundaries from {YYYY}_bilstm_merged.csv,
so each sequence step represents 15 minutes of real time.

The attention mechanism is price-volume customized:
  attn_score = LSTM_output_score + |MACD| × vol_ratio
When volume spikes alongside a strong price move (liquidation cascade),
this signal is amplified — the ACB flags it and weights that moment heavily.

Reference: arXiv 2602.12380 — TFT-ACB-XML (Din & Khan, 2026)
"""
import os, sys
import numpy as np
import joblib

ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (SEQ_LEN_BILSTM as SEQ_LEN, HORIZON_BILSTM as HORIZON,
                    N_EPOCHS_BILSTM as N_EPOCHS, BATCH_BILSTM as BATCH,
                    STRIDE_BILSTM as STRIDE, DROPOUT_BILSTM as DROPOUT,
                    N_ENSEMBLE, BILSTM_SAMPLE_INTERVAL)

HORIZON_ROWS = HORIZON // BILSTM_SAMPLE_INTERVAL  # 1440 min / 15 min-per-row = 96 rows (1 day)

MODEL_DIR      = os.path.join(ROOT, "models", "saved")
MODEL_PATH     = os.path.join(MODEL_DIR, "bilstm.keras")
SCALER_PATH    = os.path.join(MODEL_DIR, "bilstm_scaler.pkl")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints_bilstm")

def _seed_model_path(s):  return os.path.join(MODEL_DIR, f"bilstm_s{s}.keras")
def _seed_scaler_path(s): return os.path.join(MODEL_DIR, f"bilstm_scaler_s{s}.pkl")
def _seed_ckpt_dir(s):    return os.path.join(MODEL_DIR, f"checkpoints_bilstm_s{s}")
def _ensemble_ready():    return all(os.path.exists(_seed_model_path(s)) for s in range(N_ENSEMBLE))

# Feature indices within the 15m/1H feature block (24 features total)
# 15m features are first 12: m15_rsi(0), m15_macd_diff_pct(1), m15_ema9(2), ...,
#                             m15_obv_zscore(8), m15_vol_ratio(9), m15_body_ratio(10), m15_adx(11)
_MACD_IDX     = 1   # m15_macd_diff_pct — 15m price momentum proxy
_VOL_IDX      = 9   # m15_vol_ratio     — 15m volume spike indicator


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


import tensorflow as tf
from tensorflow.keras import layers, regularizers


class _SumPool(tf.keras.layers.Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)

    def get_config(self):
        return super().get_config()


class _PVAttention(tf.keras.layers.Layer):
    """
    Price-Volume customized attention.
    Weights timesteps by LSTM output relevance AND price×volume interaction,
    so liquidation cascades (high volume + strong price move) receive higher weight.
    """
    def __init__(self, macd_idx: int, vol_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.macd_idx = macd_idx
        self.vol_idx  = vol_idx
        self.dense    = layers.Dense(1, activation="tanh")

    def call(self, lstm_out, raw_inp):
        # Standard LSTM-based attention score
        attn_lstm = self.dense(lstm_out)               # [B, T, 1]

        # Price-volume interaction signal
        price_signal = tf.abs(raw_inp[..., self.macd_idx:self.macd_idx+1])  # |MACD|
        vol_signal   = raw_inp[..., self.vol_idx:self.vol_idx+1]             # vol_ratio
        pv_signal    = price_signal * vol_signal                              # [B, T, 1]

        # Combined: LSTM score amplified by price-volume signal
        attn_combined = attn_lstm + pv_signal          # [B, T, 1]
        return tf.nn.softmax(attn_combined, axis=1)    # [B, T, 1]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"macd_idx": self.macd_idx, "vol_idx": self.vol_idx})
        return cfg


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
             if f.startswith("bilstm_epoch_") and f.endswith(".keras")]
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
    return {"_SumPool": _SumPool, "_PVAttention": _PVAttention}


def build_model(n_features: int) -> tf.keras.Model:
    L2  = 1e-4
    inp = layers.Input(shape=(SEQ_LEN, n_features), name="bilstm_input")

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True,
                    kernel_regularizer=regularizers.l2(L2)),
        name="bi_lstm"
    )(inp)   # [B, T, 64]

    # Price-volume customized attention
    attn_w = _PVAttention(_MACD_IDX, _VOL_IDX, name="pv_attn")(x, inp)  # [B, T, 1]
    x      = layers.Multiply(name="attn_apply")([x, attn_w])
    x      = _SumPool(name="attn_pool")(x)   # [B, 64]

    x   = layers.Dropout(DROPOUT, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid",
                        kernel_regularizer=regularizers.l2(1e-3),
                        name="output")(x)

    model = tf.keras.models.Model(inp, out, name="acb_bilstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train(feature_df, feature_cols: list, seed: int = None) -> dict:
    import time
    from sklearn.preprocessing import StandardScaler

    if seed is not None:
        import tensorflow as _tf
        _tf.random.set_seed(seed)
        np.random.seed(seed)

    _model_path  = _seed_model_path(seed) if seed is not None else MODEL_PATH
    _scaler_path = _seed_scaler_path(seed) if seed is not None else SCALER_PATH
    _ckpt_dir    = _seed_ckpt_dir(seed)   if seed is not None else CHECKPOINT_DIR

    t0 = time.time()

    df = feature_df.dropna(subset=feature_cols).copy()
    df["target"] = (df["close"].shift(-HORIZON_ROWS) > df["close"]).astype(int)
    df = df.dropna(subset=["target"])
    print(f"  BiLSTM training rows: {len(df):,}  label balance: {df['target'].mean():.3f}",
          flush=True)

    X = df[feature_cols].values
    y = df["target"].values
    n = len(X)

    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

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

    def make_ds(X_sc, y_arr, start, end, shuffle=False):
        return tf.keras.utils.timeseries_dataset_from_array(
            X_sc,
            targets=y_arr[start + SEQ_LEN: end],
            sequence_length=SEQ_LEN,
            sequence_stride=STRIDE,
            batch_size=BATCH,
            shuffle=shuffle,
        ).prefetch(tf.data.AUTOTUNE)

    n_aug    = len(X_train)
    train_ds = make_ds(X_train, y_tr_aug, 0, n_aug, shuffle=True)
    val_ds   = make_ds(X_val,   y, train_end, val_end)
    test_ds  = make_ds(X_test,  y, val_end, n)

    os.makedirs(_ckpt_dir, exist_ok=True)
    ckpt_path, start_epoch = _latest_checkpoint(_ckpt_dir)
    n_features = len(feature_cols)
    if ckpt_path:
        try:
            loaded = tf.keras.models.load_model(ckpt_path, custom_objects=_custom_objects())
            if tuple(loaded.input_shape[1:]) == (SEQ_LEN, n_features):
                model = loaded
                print(f"  Resuming BiLSTM from checkpoint epoch {start_epoch}", flush=True)
            else:
                print(f"  Checkpoint shape mismatch — starting fresh.", flush=True)
                model = build_model(n_features)
                start_epoch = 0
        except Exception as e:
            print(f"  Checkpoint load failed ({e}) — starting fresh.", flush=True)
            model = build_model(n_features)
            start_epoch = 0
    else:
        model = build_model(n_features)
        start_epoch = 0

    print(f"  BiLSTM parameters: {model.count_params():,}", flush=True)

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
        _CheckpointCB(_ckpt_dir, "bilstm", every_n=5),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=N_EPOCHS, initial_epoch=start_epoch,
                        callbacks=callbacks, verbose=0)

    val_acc  = max(history.history.get("val_accuracy", [0.5]))
    _, test_acc = model.evaluate(test_ds, verbose=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(_model_path)
    joblib.dump(scaler, _scaler_path)
    if seed is None:
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    return {"test_acc": float(test_acc), "val_acc": float(val_acc),
            "time_taken": time.time() - t0}


def train_ensemble(feature_df, feature_cols: list) -> list:
    """Train N_ENSEMBLE seeds and return list of results."""
    results = []
    for s in range(N_ENSEMBLE):
        print(f"\n{'='*50}", flush=True)
        print(f"BiLSTM ENSEMBLE — seed {s+1}/{N_ENSEMBLE}", flush=True)
        print(f"{'='*50}", flush=True)
        r = train(feature_df, feature_cols, seed=s)
        results.append(r)
        print(f"  Seed {s}: val={r['val_acc']:.4f}  test={r['test_acc']:.4f}", flush=True)
    avg_val  = float(np.mean([r["val_acc"]  for r in results]))
    avg_test = float(np.mean([r["test_acc"] for r in results]))
    print(f"\n  Ensemble average: val={avg_val:.4f}  test={avg_test:.4f}", flush=True)
    return results


def _infer_batch_single(model, scaler, X_all, batch_size):
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


def predict_proba(X_seq: np.ndarray, feature_cols: list = None) -> float:
    X_in = X_seq[-SEQ_LEN:]
    if _ensemble_ready():
        preds = []
        for s in range(N_ENSEMBLE):
            m  = tf.keras.models.load_model(_seed_model_path(s), custom_objects=_custom_objects())
            sc = joblib.load(_seed_scaler_path(s))
            inp = sc.transform(X_in).astype(np.float32).reshape(1, SEQ_LEN, -1)
            preds.append(float(m.predict(inp, verbose=0)[0][0]))
            del m
        return float(np.mean(preds))
    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)
    inp    = scaler.transform(X_in).astype(np.float32).reshape(1, SEQ_LEN, -1)
    return float(model.predict(inp, verbose=0)[0][0])


def predict_proba_batch(X_all: np.ndarray, feature_cols: list = None,
                         batch_size: int = 512) -> np.ndarray:
    if _ensemble_ready():
        all_probs = []
        for s in range(N_ENSEMBLE):
            m  = tf.keras.models.load_model(_seed_model_path(s), custom_objects=_custom_objects())
            sc = joblib.load(_seed_scaler_path(s))
            all_probs.append(_infer_batch_single(m, sc, X_all, batch_size))
            del m
        return np.mean(all_probs, axis=0)
    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)
    return _infer_batch_single(model, scaler, X_all, batch_size)


def is_trained() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
