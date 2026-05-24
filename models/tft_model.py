"""
Temporal Fusion Transformer — macro direction model (Lim et al. 2021).

Predicts 1-day price direction using 30 days of 4h-sampled context.
Features: h4/d1 indicators (24 features, all dynamic — no static covariates).
Input data comes pre-sampled at 4-hour boundaries from {YYYY}_tft_merged.csv,
so no internal downsampling is needed.

Architecture: true TFT with VSN on past+future, encoder-decoder LSTM,
GRN gated skips, temporal self-attention over [past; future], quantile heads.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (SEQ_LEN_TFT as SEQ_LEN, HORIZON_TFT as HORIZON,
                    N_EPOCHS_TFT as N_EPOCHS, BATCH_TFT as BATCH,
                    STRIDE_TFT as STRIDE, D_MODEL, N_HEADS, DROPOUT_TFT as DROPOUT,
                    N_ENSEMBLE, TFT_SAMPLE_INTERVAL, EXP_WEIGHT_HALFLIFE)

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


class _LSTMDecoder(tf.keras.layers.Layer):
    """Single-step LSTM decoder initialized from encoder final states."""
    def __init__(self, units, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self._units   = units
        self._dropout = dropout
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True,
                                 kernel_regularizer=regularizers.l2(1e-4))
        self.drop = layers.Dropout(dropout)

    def call(self, future_proj, initial_state, training=False):
        out, _, _ = self.lstm(future_proj, initial_state=initial_state,
                               training=training)
        return self.drop(out, training=training)  # [B, 1, units]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self._units, "dropout": self._dropout})
        return cfg


class _Q50Accuracy(tf.keras.metrics.Metric):
    """Binary accuracy using the q50 head (column 1 of the 3-quantile output)."""
    def __init__(self, **kwargs):
        super().__init__(name="q50_acc", **kwargs)
        self._total   = self.add_weight("total",   initializer="zeros")
        self._correct = self.add_weight("correct", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        q50   = y_pred[:, 1]
        pred  = tf.cast(q50 > 0.5, tf.int32)
        true_ = tf.cast(tf.reshape(y_true, (-1,)) > 0.5, tf.int32)
        n_ok  = tf.cast(tf.reduce_sum(tf.cast(pred == true_, tf.float32)), tf.float32)
        self._total.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))
        self._correct.assign_add(n_ok)

    def result(self):
        return self._correct / (self._total + 1e-7)

    def reset_state(self):
        self._total.assign(0.0)
        self._correct.assign(0.0)

    def get_config(self):
        return super().get_config()


def _pinball_loss(quantiles=(0.1, 0.5, 0.9)):
    """Pinball/quantile loss summed over three heads. y_true: scalar, y_pred: [B, 3]."""
    qs = tf.constant(list(quantiles), dtype=tf.float32)
    def loss(y_true, y_pred):
        y = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32)
        e = y - y_pred
        return tf.reduce_mean(tf.maximum(qs * e, (qs - 1.0) * e))
    loss.__name__ = "pinball_loss"
    return loss


def encode_time_features(timestamps: "pd.Series",
                          horizon_minutes: int = HORIZON) -> np.ndarray:
    """6 cyclical features for the predicted step (t + horizon). Shape [n, 6]."""
    pred_ts = timestamps + pd.Timedelta(minutes=horizon_minutes)
    sin_h   = np.sin(2 * np.pi * pred_ts.dt.hour        / 24)
    cos_h   = np.cos(2 * np.pi * pred_ts.dt.hour        / 24)
    sin_dow = np.sin(2 * np.pi * pred_ts.dt.dayofweek   / 7)
    cos_dow = np.cos(2 * np.pi * pred_ts.dt.dayofweek   / 7)
    sin_mon = np.sin(2 * np.pi * pred_ts.dt.month       / 12)
    cos_mon = np.cos(2 * np.pi * pred_ts.dt.month       / 12)
    return np.stack([sin_h, cos_h, sin_dow, cos_dow, sin_mon, cos_mon],
                    axis=1).astype(np.float32)


def _make_tft_ds(X_sc: np.ndarray, X_time: np.ndarray, y_arr: np.ndarray,
                  abs_start: int, seq_len: int, stride: int, batch_size: int,
                  w_arr: np.ndarray = None) -> "tf.data.Dataset":
    """
    Two-input TFT dataset: yields ((past [SEQ_LEN, n_dyn], fut [1, 6]), target[, weight]).
    X_sc:      scaled features for this split, shape [n_local, n_dyn]
    X_time:    time features for ALL rows, shape [n_total, 6]
    y_arr:     labels for ALL rows, shape [n_total]
    abs_start: absolute row index where X_sc begins (0 for train, train_end for val, etc.)
    w_arr:     sample weights for [n_local] training rows; None → no weight output
    """
    n_local = len(X_sc)
    starts  = np.arange(0, n_local - seq_len, stride, dtype=np.int32)
    tgts    = y_arr[abs_start + starts + seq_len].astype(np.float32)
    X_tf    = tf.constant(X_sc,   dtype=tf.float32)
    Xt_tf   = tf.constant(X_time, dtype=tf.float32)

    if w_arr is not None:
        wts = w_arr[starts + seq_len - 1].astype(np.float32)
        ds  = tf.data.Dataset.from_tensor_slices((starts, tgts, wts))
        ds  = ds.shuffle(buffer_size=min(len(starts), 10000),
                         reshuffle_each_iteration=True)
        def _extract_w(s, tgt, wt):
            past = tf.slice(X_tf,  [s, 0], [seq_len, -1])
            fut  = tf.slice(Xt_tf, [abs_start + s + seq_len, 0], [1, 6])
            return (past, fut), tgt, wt
        return (ds.map(_extract_w, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    else:
        ds = tf.data.Dataset.from_tensor_slices((starts, tgts))
        def _extract(s, tgt):
            past = tf.slice(X_tf,  [s, 0], [seq_len, -1])
            fut  = tf.slice(Xt_tf, [abs_start + s + seq_len, 0], [1, 6])
            return (past, fut), tgt
        return (ds.map(_extract, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(batch_size).prefetch(tf.data.AUTOTUNE))


def build_model(n_dynamic: int, n_static: int = 0,
                d_model: int = None, dropout_rate: float = None) -> tf.keras.Model:
    """
    True Temporal Fusion Transformer (Lim et al. 2021).

    Architecture:
      1. VSN on BOTH past and future inputs
      2. Encoder-Decoder LSTM (encoder processes past, decoder gets future + encoder state)
      3. GRN gated skip connections at LSTM outputs (residual back to VSN)
      4. Temporal SELF-attention over full [past_enriched; fut_enriched] context (SEQ_LEN+1 steps)
      5. GRN gated skip after attention; GRN position-wise FFN
      6. Extract LAST position (future step) from output
      7. Three quantile heads → [B, 3]  (q10, q50, q90)

    Two inputs: past_inp [B, SEQ_LEN, n_dynamic], fut_inp [B, 1, 6].
    Output: [B, 3].  Trained with pinball loss.
    n_static is accepted but unused (no static covariate path in current data).
    """
    from tensorflow.keras import models

    dm = d_model      if d_model      is not None else D_MODEL
    dr = dropout_rate if dropout_rate is not None else DROPOUT

    # ── Inputs ────────────────────────────────────────────────────────────────
    past_inp = layers.Input(shape=(SEQ_LEN, n_dynamic), name="past_inp")
    fut_inp  = layers.Input(shape=(1, 6),               name="fut_inp")

    # ── 1. VSN on both inputs ─────────────────────────────────────────────────
    past_vsn, _ = VSN(n_dynamic, dm, dr, name="past_vsn")(past_inp)  # [B, SEQ_LEN, dm]
    fut_vsn,  _ = VSN(6, dm, dr, name="fut_vsn")(fut_inp)            # [B, 1, dm]

    # ── 2. Encoder-Decoder LSTM ───────────────────────────────────────────────
    enc_lstm = layers.LSTM(dm, return_sequences=True, return_state=True,
                            kernel_regularizer=regularizers.l2(1e-4),
                            name="lstm_enc")
    enc_out, enc_h, enc_c = enc_lstm(past_vsn)                       # [B, SEQ_LEN, dm]
    dec_out = _LSTMDecoder(dm, dr, name="lstm_dec")(
                  fut_vsn, initial_state=[enc_h, enc_c])              # [B, 1, dm]

    # ── 3. GRN gated skip connections (Lim et al. §4.2) ──────────────────────
    # Residual skips back to the VSN output — the "gated residual connection" in the paper.
    enc_enriched = layers.LayerNormalization(name="ln_enc")(
        GRN(dm, dr, name="grn_enc")(enc_out) + past_vsn)             # [B, SEQ_LEN, dm]
    dec_enriched = layers.LayerNormalization(name="ln_dec")(
        GRN(dm, dr, name="grn_dec")(dec_out) + fut_vsn)              # [B, 1, dm]

    # ── 4. Temporal SELF-attention over full [past; future] context ───────────
    temporal = layers.Concatenate(axis=1, name="temporal_concat")(
        [enc_enriched, dec_enriched])                                 # [B, SEQ_LEN+1, dm]
    attn_out = layers.MultiHeadAttention(
        num_heads=N_HEADS, key_dim=max(1, dm // N_HEADS),
        dropout=dr, name="temporal_attn",
    )(query=temporal, value=temporal, key=temporal)                   # [B, SEQ_LEN+1, dm]

    # GRN gated skip after attention
    attn_enriched = layers.LayerNormalization(name="ln_attn")(
        GRN(dm, dr, name="grn_attn")(attn_out) + temporal)           # [B, SEQ_LEN+1, dm]

    # ── 5. Position-wise GRN FFN ─────────────────────────────────────────────
    ffn_enriched = layers.LayerNormalization(name="ln_ffn")(
        GRN(dm, dr, name="grn_ffn")(attn_enriched) + attn_enriched)  # [B, SEQ_LEN+1, dm]

    # ── 6. Extract LAST position (future decoder step) ───────────────────────
    x = layers.Lambda(lambda t: t[:, -1, :], name="extract_future")(ffn_enriched)  # [B, dm]

    # ── 7. Three quantile heads ───────────────────────────────────────────────
    q10 = layers.Dense(1, name="q10")(x)
    q50 = layers.Dense(1, name="q50")(x)
    q90 = layers.Dense(1, name="q90")(x)
    out = layers.Concatenate(name="quantiles")([q10, q50, q90])       # [B, 3]

    model = models.Model([past_inp, fut_inp], out, name="tft")
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
    return {"GRN": GRN, "VSN": VSN, "_LSTMDecoder": _LSTMDecoder,
            "_Q50Accuracy": _Q50Accuracy}


# Diverse per-seed hyperparameters — architectural variation improves ensemble signal coverage
_TFT_SEED_CFG = [
    {"d_model": 48,  "dropout": 0.20, "lr": 5e-4},
    {"d_model": 64,  "dropout": 0.30, "lr": 1e-3},
    {"d_model": 96,  "dropout": 0.40, "lr": 2e-3},
]


def _compute_exp_weights(n: int, y: np.ndarray,
                          halflife_frac: float = EXP_WEIGHT_HALFLIFE) -> np.ndarray:
    """Exp decay weights × class correction, normalized to mean=1."""
    lam   = np.log(2) / (halflife_frac * n)
    exp_w = np.exp(-lam * (n - 1 - np.arange(n)))
    pos   = float(y.mean())
    neg   = 1.0 - pos
    cf    = np.where(y == 1, 1.0 / (2.0 * max(pos, 1e-6)),
                              1.0 / (2.0 * max(neg, 1e-6)))
    combined = (exp_w * cf).astype(np.float32)
    return combined / combined.mean()


def train(feature_df, dynamic_cols: list, static_cols: list,
          seed: int = None, seed_cfg: dict = None) -> dict:
    import time
    from sklearn.preprocessing import StandardScaler

    seed_cfg   = seed_cfg or {}
    _d_model   = seed_cfg.get("d_model",  D_MODEL)
    _dropout   = seed_cfg.get("dropout",  DROPOUT)
    _lr_init   = seed_cfg.get("lr",       1e-3)

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    _model_path  = _seed_model_path(seed) if seed is not None else MODEL_PATH
    _scaler_path = _seed_scaler_path(seed) if seed is not None else SCALER_PATH
    _ckpt_dir    = _seed_ckpt_dir(seed)   if seed is not None else CHECKPOINT_DIR

    t0 = time.time()

    all_cols  = dynamic_cols + static_cols
    df = feature_df.dropna(subset=all_cols).copy()
    if "target" not in df.columns:
        raise RuntimeError("target column missing — call apply_triple_barrier in train.py first")

    n_dynamic = len(dynamic_cols)

    # Input is already 4h-sampled from tft_merged — no downsampling needed here.
    print(f"  TFT training rows: {len(df):,} (4h-sampled)  label balance: {df['target'].mean():.3f}  "
          f"dynamic={n_dynamic}", flush=True)

    X = df[dynamic_cols].values   # [n, n_dynamic]  (no static in current data)
    y = df["target"].values
    n = len(X)

    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    # Exponential decay weights (Fix 3) — replaces recency augmentation
    y_tr = y[:train_end]
    w_tr = _compute_exp_weights(train_end, y_tr)
    print(f"  Exp weights: halflife={EXP_WEIGHT_HALFLIFE:.0%}  label={y_tr.mean():.3f}",
          flush=True)

    # Time features for TFT decoder future input (Fix 4)
    if "time" not in df.columns:
        X_time = np.zeros((n, 6), dtype=np.float32)
    else:
        X_time = encode_time_features(df["time"].reset_index(drop=True))  # [n, 6]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[:train_end]).astype(np.float32)
    X_val   = scaler.transform(X[train_end:val_end]).astype(np.float32)
    X_test  = scaler.transform(X[val_end:]).astype(np.float32)

    train_ds = _make_tft_ds(X_train, X_time, y, 0,         SEQ_LEN, STRIDE, BATCH, w_arr=w_tr)
    val_ds   = _make_tft_ds(X_val,   X_time, y, train_end, SEQ_LEN, STRIDE, BATCH)
    test_ds  = _make_tft_ds(X_test,  X_time, y, val_end,   SEQ_LEN, STRIDE, BATCH)

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    os.makedirs(_ckpt_dir, exist_ok=True)
    ckpt_path, start_epoch = _latest_checkpoint(_ckpt_dir)
    if ckpt_path:
        try:
            loaded = tf.keras.models.load_model(ckpt_path, custom_objects=_custom_objects())
            shapes = loaded.input_shape
            ok = (isinstance(shapes, list) and len(shapes) == 2 and
                  tuple(shapes[0][1:]) == (SEQ_LEN, n_dynamic) and
                  tuple(shapes[1][1:]) == (1, 6))
            if ok:
                model = loaded
                lbl = f"seed {seed}" if seed is not None else "TFT"
                print(f"  Resuming {lbl} from checkpoint epoch {start_epoch}", flush=True)
            else:
                print(f"  Checkpoint shape mismatch — starting fresh.", flush=True)
                model = build_model(n_dynamic, d_model=_d_model, dropout_rate=_dropout)
                start_epoch = 0
        except Exception as e:
            print(f"  Checkpoint load failed ({e}) — starting fresh.", flush=True)
            model = build_model(n_dynamic, d_model=_d_model, dropout_rate=_dropout)
            start_epoch = 0
    else:
        model = build_model(n_dynamic, d_model=_d_model, dropout_rate=_dropout)
        start_epoch = 0

    print(f"  Parameters: {model.count_params():,}", flush=True)

    steps_per_epoch = max(1, train_end // BATCH)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=_lr_init,
        decay_steps=N_EPOCHS * steps_per_epoch,
        alpha=1e-5,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0),
        loss=_pinball_loss(),
        metrics=[_Q50Accuracy()],
    )

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
                f"q50 acc: {logs.get('q50_acc',0):.3f} | "
                f"Val q50: {logs.get('val_q50_acc',0):.3f} | "
                f"Elapsed: {_fmt(elapsed)} | ETA: {_fmt(eta)}",
                flush=True,
            )

    callbacks = [
        _ProgressCB(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, mode="min", restore_best_weights=True),
        _CheckpointCB(_ckpt_dir, "tft", every_n=5),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=N_EPOCHS, initial_epoch=start_epoch,
                        callbacks=callbacks, verbose=0)

    val_acc = max(history.history.get("val_q50_acc", [0.5]))

    # Test accuracy from q50 head
    raw_test = model.predict(test_ds, verbose=0)    # [N_seqs, 3]
    q50_test = raw_test[:, 1]
    n_test_seqs = len(q50_test)
    y_test_aligned = y[val_end + SEQ_LEN : val_end + SEQ_LEN + n_test_seqs * STRIDE : STRIDE]
    y_test_aligned = y_test_aligned[:n_test_seqs]
    test_acc = float(np.mean((q50_test > 0.5).astype(int) == y_test_aligned))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(_model_path)
    joblib.dump(scaler, _scaler_path)
    if seed is None:
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    return {"test_acc": float(test_acc), "val_acc": float(val_acc),
            "time_taken": time.time() - t0}


def train_ensemble(feature_df, dynamic_cols: list, static_cols: list) -> list:
    """Train N_ENSEMBLE seeds with diverse hyperparameters. Predictions averaged at inference."""
    results = []
    for s in range(N_ENSEMBLE):
        cfg = _TFT_SEED_CFG[s] if s < len(_TFT_SEED_CFG) else {}
        print(f"\n{'='*50}", flush=True)
        print(f"TFT ENSEMBLE — seed {s+1}/{N_ENSEMBLE}  "
              f"d_model={cfg.get('d_model', D_MODEL)}  "
              f"dropout={cfg.get('dropout', DROPOUT)}  "
              f"lr={cfg.get('lr', 1e-3)}", flush=True)
        print(f"{'='*50}", flush=True)
        r = train(feature_df, dynamic_cols, static_cols, seed=s, seed_cfg=cfg)
        results.append(r)
        print(f"  Seed {s}: val={r['val_acc']:.4f}  test={r['test_acc']:.4f}", flush=True)
    avg_val  = float(np.mean([r["val_acc"]  for r in results]))
    avg_test = float(np.mean([r["test_acc"] for r in results]))
    print(f"\n  Ensemble average: val={avg_val:.4f}  test={avg_test:.4f}", flush=True)
    return results


# ── Inference ─────────────────────────────────────────────────────────────────

def _infer_batch(model, scaler, X_dynamic, timestamps, batch_size):
    """
    Batch inference for one TFT model.
    Returns q50 probabilities (float32 array, same length as X_dynamic).
    timestamps: pd.Series or array-like of UTC timestamps for X_dynamic rows.
    """
    X_sc = scaler.transform(X_dynamic).astype(np.float32)
    n    = len(X_sc)
    probs = np.full(n, 0.5, dtype=np.float32)

    if timestamps is not None:
        X_time = encode_time_features(pd.Series(timestamps))
    else:
        X_time = np.zeros((n, 6), dtype=np.float32)

    n_batches = max(1, (n - SEQ_LEN + batch_size - 1) // batch_size)
    for b in range(n_batches):
        b_start = b * batch_size
        b_end   = min(b_start + batch_size, n - SEQ_LEN)
        if b_start >= b_end:
            break
        abs_idx    = np.arange(SEQ_LEN + b_start, SEQ_LEN + b_end)
        win_idx    = abs_idx[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
        past_batch = X_sc[win_idx]                       # [B, SEQ_LEN, n_dyn]
        fut_batch  = X_time[abs_idx][:, np.newaxis, :]  # [B, 1, 6]
        raw        = model.predict([past_batch, fut_batch], verbose=0,
                                    batch_size=batch_size)  # [B, 3]
        # Anti-crossing clamp
        q50 = np.clip(raw[:, 1], raw[:, 0], raw[:, 2])
        probs[SEQ_LEN + b_start: SEQ_LEN + b_end] = q50
    return probs


def predict_proba_batch(X_dynamic: np.ndarray, X_static: np.ndarray,
                         timestamps=None, batch_size: int = 512) -> np.ndarray:
    """
    Batch inference. Auto-uses ensemble (N_ENSEMBLE seeds averaged) if trained.
    timestamps: optional pd.Series or array of UTC row timestamps for decoder future input.
    X_static is accepted but unused (no static covariate path in current architecture).
    """
    if _ensemble_ready():
        all_probs = []
        for s in range(N_ENSEMBLE):
            m  = tf.keras.models.load_model(_seed_model_path(s), custom_objects=_custom_objects())
            sc = joblib.load(_seed_scaler_path(s))
            all_probs.append(_infer_batch(m, sc, X_dynamic, timestamps, batch_size))
            del m
        return np.mean(all_probs, axis=0)

    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)
    return _infer_batch(model, scaler, X_dynamic, timestamps, batch_size)


def predict_proba(X_dynamic: np.ndarray, X_static: np.ndarray,
                  timestamps=None) -> float:
    """
    Single-sample inference. Returns q50 as a scalar probability.
    X_dynamic must have at least SEQ_LEN rows (already 4h-sampled).
    timestamps: optional pd.Series or array for decoder future input;
                if None, uses current UTC time.
    """
    X_h = X_dynamic[-SEQ_LEN:]

    if timestamps is not None:
        last_ts = pd.Series(timestamps).iloc[-1:]
        fut_feat = encode_time_features(last_ts)  # [1, 6]
    else:
        now = pd.Timestamp.now(tz="UTC")
        fut_feat = encode_time_features(pd.Series([now]))  # [1, 6]
    fut = fut_feat[np.newaxis, :, :]  # [1, 1, 6]

    if _ensemble_ready():
        preds = []
        for s in range(N_ENSEMBLE):
            m  = tf.keras.models.load_model(_seed_model_path(s), custom_objects=_custom_objects())
            sc = joblib.load(_seed_scaler_path(s))
            past = sc.transform(X_h).astype(np.float32).reshape(1, SEQ_LEN, -1)
            raw  = m.predict([past, fut], verbose=0)  # [1, 3]
            preds.append(float(np.clip(raw[0, 1], raw[0, 0], raw[0, 2])))
            del m
        return float(np.mean(preds))

    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)
    past   = scaler.transform(X_h).astype(np.float32).reshape(1, SEQ_LEN, -1)
    raw    = model.predict([past, fut], verbose=0)
    return float(np.clip(raw[0, 1], raw[0, 0], raw[0, 2]))


def predict_uncertainty(X_dynamic: np.ndarray, X_static: np.ndarray,
                         timestamps=None) -> tuple:
    """Returns (q10, q50, q90) for uncertainty-aware signal generation (Fix 5)."""
    X_h = X_dynamic[-SEQ_LEN:]

    if timestamps is not None:
        last_ts = pd.Series(timestamps).iloc[-1:]
        fut_feat = encode_time_features(last_ts)
    else:
        fut_feat = encode_time_features(pd.Series([pd.Timestamp.now(tz="UTC")]))
    fut = fut_feat[np.newaxis, :, :]

    if _ensemble_ready():
        qs_list = []
        for s in range(N_ENSEMBLE):
            m  = tf.keras.models.load_model(_seed_model_path(s), custom_objects=_custom_objects())
            sc = joblib.load(_seed_scaler_path(s))
            past = sc.transform(X_h).astype(np.float32).reshape(1, SEQ_LEN, -1)
            raw  = m.predict([past, fut], verbose=0)[0]  # [3]
            qs_list.append(raw)
            del m
        avg = np.mean(qs_list, axis=0)
        return (float(avg[0]), float(np.clip(avg[1], avg[0], avg[2])), float(avg[2]))

    model  = tf.keras.models.load_model(MODEL_PATH, custom_objects=_custom_objects())
    scaler = joblib.load(SCALER_PATH)
    past   = scaler.transform(X_h).astype(np.float32).reshape(1, SEQ_LEN, -1)
    raw    = model.predict([past, fut], verbose=0)[0]
    return (float(raw[0]), float(np.clip(raw[1], raw[0], raw[2])), float(raw[2]))


def is_trained() -> bool:
    return _ensemble_ready() or (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH))
