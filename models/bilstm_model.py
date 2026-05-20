"""
Attention-Customized Bidirectional LSTM (ACB) — short-term momentum model.

Inputs:  1m, 15m, 30m features only (30 cols) — high-velocity local signals
Context: 4 hours of 1-minute candles (SEQ_LEN=240)
Target:  1-hour price direction (HORIZON=60)

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
                    STRIDE_BILSTM as STRIDE, DROPOUT_BILSTM as DROPOUT)

MODEL_DIR      = os.path.join(ROOT, "models", "saved")
MODEL_PATH     = os.path.join(MODEL_DIR, "bilstm.keras")
SCALER_PATH    = os.path.join(MODEL_DIR, "bilstm_scaler.pkl")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints_bilstm")

# Feature indices within the 1m/15m/30m feature block (30 features total)
# 1m features are first 10: rsi(0), macd_diff_pct(1), ema9(2), ema21(3), ema50(4),
#                            atr_norm(5), bb_pct(6), bb_width(7), obv_zscore(8), vol_ratio(9)
_MACD_IDX     = 1   # 1m macd_diff_pct — price momentum proxy
_VOL_IDX      = 9   # 1m vol_ratio    — volume spike indicator


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


def _latest_checkpoint() -> tuple:
    if not os.path.exists(CHECKPOINT_DIR):
        return None, 0
    files = [f for f in os.listdir(CHECKPOINT_DIR)
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
    return (os.path.join(CHECKPOINT_DIR, best), best_ep) if best else (None, 0)


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


def train(feature_df, feature_cols: list) -> dict:
    import time
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()

    df = feature_df.dropna(subset=feature_cols).copy()
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(int)
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

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path, start_epoch = _latest_checkpoint()
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

    BAR = 30

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
        _CheckpointCB(CHECKPOINT_DIR, "bilstm", every_n=5),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=N_EPOCHS, initial_epoch=start_epoch,
                        callbacks=callbacks, verbose=0)

    val_acc  = max(history.history.get("val_accuracy", [0.5]))
    _, test_acc = model.evaluate(test_ds, verbose=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return {"test_acc": float(test_acc), "val_acc": float(val_acc),
            "time_taken": time.time() - t0}


def predict_proba(X_seq: np.ndarray, feature_cols: list = None) -> float:
    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)
    Xs  = scaler.transform(X_seq).astype(np.float32)
    inp = Xs[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)
    return float(model.predict(inp, verbose=0)[0][0])


def predict_proba_batch(X_all: np.ndarray, feature_cols: list = None,
                         batch_size: int = 512) -> np.ndarray:
    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)

    X_sc  = scaler.transform(X_all).astype(np.float32)
    n     = len(X_sc)
    probs = np.full(n, 0.5, dtype=np.float32)

    n_batches = (n - SEQ_LEN + batch_size - 1) // batch_size
    for b in range(n_batches):
        b_start = b * batch_size
        b_end   = min(b_start + batch_size, n - SEQ_LEN)
        abs_idx = np.arange(SEQ_LEN + b_start, SEQ_LEN + b_end)
        win_idx = abs_idx[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
        batch_X = X_sc[win_idx]
        preds   = model.predict(batch_X, verbose=0, batch_size=batch_size)
        probs[SEQ_LEN + b_start: SEQ_LEN + b_end] = preds.flatten()

    return probs


def is_trained() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
