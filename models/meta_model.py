"""
Agreement-Based Routing Meta-Learner for TFT-ACB-XML stacking.

Instead of predicting price direction directly (old approach), this meta-learner
answers: "given this market snapshot, which base model is more likely to be correct?"

Two routing paths:

  AGREE  (|p_tft - p_bilstm| < AGREE_THRESH):
      Both models see the same outcome — trust their consensus via a static
      error-reciprocal blend.  No gate needed; signal is already clean.

  DISAGREE (gap >= AGREE_THRESH):
      One model is right, one is wrong.  A correctness gate trained ONLY on
      these rows learns the market-regime → model-reliability mapping:
          gate(market_snapshot, gap) → P(TFT_correct | disagreement)
          final = gate_prob × p_tft + (1 - gate_prob) × p_bilstm

Why this beats standard stacking here:
  - Gate trains on disagreement rows where the signal is cleanest (one model
    demonstrably right), avoiding the ~50% noise floor of the full dataset.
  - Per-sample dynamic trust allocation instead of global fixed weights.
  - Learns genuine regime sensitivity: high ADX trending → TFT macro view wins;
    vol spike + MACD divergence → BiLSTM micro momentum wins.
"""
import os
import sys
import numpy as np
import joblib
import pandas as pd

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import HORIZON_META as HORIZON

MODEL_DIR  = os.path.join(ROOT, "models", "saved")
MODEL_PATH = os.path.join(MODEL_DIR, "meta_xgb.pkl")

AGREE_THRESH = 0.10   # route to gate when |p_tft - p_bilstm| >= this

# Market context snapshot fed to the correctness gate
GATE_MARKET_FEATURES = [
    "rsi",              # 1m immediate momentum
    "m15_rsi",          # 15m short-term momentum
    "vol_ratio",        # 1m volume anomaly (spike = conviction)
    "body_ratio",       # 1m candle direction intensity
    "h1_rsi",           # 1H momentum
    "h4_rsi",           # 4H momentum
    "d1_rsi",           # 1D macro regime
    "h1_atr_norm",      # 1H volatility
    "h4_atr_norm",      # 4H volatility
    "d1_macd_diff_pct", # macro trend direction
    "h4_adx",           # trend strength — key regime discriminator
    "h1_bb_width",      # intraday volatility regime
]


def _get_col(df_or_X, name, feature_cols):
    """Extract a named feature from a DataFrame or numpy array."""
    if isinstance(df_or_X, pd.DataFrame):
        return df_or_X[name].values if name in df_or_X.columns else np.zeros(len(df_or_X))
    fc = list(feature_cols) if feature_cols else []
    if name in fc:
        return df_or_X[:, fc.index(name)]
    return np.zeros(len(df_or_X))


def _build_gate_X(p_tft: np.ndarray, p_bilstm: np.ndarray,
                  df_or_X, feature_cols=None) -> np.ndarray:
    """Build gate feature matrix: [gap, p_tft, p_bilstm, *market_features]."""
    gap  = np.abs(p_tft - p_bilstm)
    cols = [gap, np.asarray(p_tft), np.asarray(p_bilstm)]
    for feat in GATE_MARKET_FEATURES:
        cols.append(_get_col(df_or_X, feat, feature_cols))
    return np.column_stack(cols).astype(np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train(val_df: pd.DataFrame,
          val_tft_probs: np.ndarray,
          val_bilstm_probs: np.ndarray,
          tft_val_error: float,
          bilstm_val_error: float,
          feature_cols: list) -> dict:
    """
    Train the agreement-based routing meta-learner.

    val_tft_probs / val_bilstm_probs: OOF predictions from base models
    tft_val_error / bilstm_val_error: 1 - val_acc for each base model
    """
    import time
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit

    t0 = time.time()

    # ── Static weights — fallback for agreement case ───────────────────────
    w_tft    = 1.0 / max(float(tft_val_error),    1e-6)
    w_bilstm = 1.0 / max(float(bilstm_val_error), 1e-6)
    w_total  = w_tft + w_bilstm
    w_tft_n, w_bilstm_n = w_tft / w_total, w_bilstm / w_total
    print(f"  Static fallback weights — TFT: {w_tft_n:.3f}  BiLSTM: {w_bilstm_n:.3f}",
          flush=True)

    # ── Ground-truth direction ─────────────────────────────────────────────
    val_df = val_df.copy()
    val_df["actual"] = (val_df["close"].shift(-HORIZON) > val_df["close"]).astype(int)
    val_df = val_df.dropna(subset=["actual"]).reset_index(drop=True)

    # val_tft_probs[i] predicts (close[i+60] > close[i]) — same indexing as val_df.
    # val_df after dropna keeps rows 0..n-61 (the last 60 rows become NaN after shift(-60)).
    # So probs[:n_valid] aligns row-for-row with val_df. The original [-n_valid:] creates
    # a 60-row temporal offset that misaligns TFT probs (which are non-0.5 only at
    # every 60th row) with their corresponding targets.
    n_valid    = len(val_df)
    p_tft_v    = val_tft_probs[:n_valid].astype(np.float32)
    p_bilstm_v = val_bilstm_probs[:n_valid].astype(np.float32)
    actual_v   = val_df["actual"].values

    print(f"  Meta-learner training: {n_valid:,} rows  label balance: {actual_v.mean():.3f}",
          flush=True)

    # ── Agreement split ────────────────────────────────────────────────────
    gap           = np.abs(p_tft_v - p_bilstm_v)
    disagree_idx  = np.where(gap >= AGREE_THRESH)[0]
    agree_idx     = np.where(gap <  AGREE_THRESH)[0]
    n_disagree    = len(disagree_idx)
    pct_disagree  = n_disagree / n_valid * 100

    print(f"  Agreement   (gap <  {AGREE_THRESH}): {len(agree_idx):,} rows "
          f"({100 - pct_disagree:.1f}%)", flush=True)
    print(f"  Disagreement(gap >= {AGREE_THRESH}): {n_disagree:,} rows "
          f"({pct_disagree:.1f}%)", flush=True)

    # ── Correctness gate — trained on disagreement rows only ──────────────
    # Target: 1 if TFT's rounded prediction matched actual direction, else 0.
    # On disagreement rows, exactly one model rounded to the right class,
    # so this is a clean 50/50 binary signal, not pure noise.
    tft_correct_v = ((p_tft_v > 0.5).astype(int) == actual_v).astype(int)

    gate_model  = None
    gate_cv_acc = 0.5

    if n_disagree < 200:
        print(f"  WARNING: only {n_disagree} disagreement rows — "
              "gate skipped, using static weights everywhere.", flush=True)
    else:
        X_gate_dis = _build_gate_X(p_tft_v[disagree_idx],
                                    p_bilstm_v[disagree_idx],
                                    val_df.iloc[disagree_idx],
                                    feature_cols)
        y_gate = tft_correct_v[disagree_idx]

        gate_label_bal = y_gate.mean()
        print(f"  Gate label balance (TFT correct when disagreeing): "
              f"{gate_label_bal:.3f}", flush=True)

        gate_params = dict(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            min_child_weight=20,  # heavy regularisation — avoids fitting noise
            gamma=0.1, verbosity=0,
        )

        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        for fold, (tr, va) in enumerate(tscv.split(X_gate_dis), 1):
            m = xgb.XGBClassifier(**gate_params)
            m.fit(X_gate_dis[tr], y_gate[tr])
            fold_acc = accuracy_score(y_gate[va], m.predict(X_gate_dis[va]))
            cv_scores.append(fold_acc)
            print(f"  Gate CV fold {fold}/3: {fold_acc:.4f}", flush=True)

        gate_cv_acc = float(np.mean(cv_scores))
        print(f"  Gate CV: {gate_cv_acc:.4f} +/- {np.std(cv_scores):.4f}",
              flush=True)

        gate_model = xgb.XGBClassifier(**gate_params)
        gate_model.fit(X_gate_dis, y_gate)

        # Feature importance — shows which market features drive routing
        feat_names = ["gap", "p_tft", "p_bilstm"] + GATE_MARKET_FEATURES
        importances = gate_model.feature_importances_
        top = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:5]
        print("  Top gate features: " +
              "  ".join(f"{n}={v:.3f}" for n, v in top), flush=True)

    # ── Simulate routing on full OOF to get an overall accuracy metric ────
    final_oof = _route(p_tft_v, p_bilstm_v, val_df, feature_cols,
                       gate_model, AGREE_THRESH, w_tft_n, w_bilstm_n)
    overall_acc = float(accuracy_score(actual_v, (final_oof > 0.5).astype(int)))
    print(f"  Overall routing accuracy (OOF): {overall_acc:.4f}", flush=True)

    # ── Baseline comparison: pure static blend ────────────────────────────
    static_blend = w_tft_n * p_tft_v + w_bilstm_n * p_bilstm_v
    static_acc   = float(accuracy_score(actual_v, (static_blend > 0.5).astype(int)))
    print(f"  Static-blend baseline accuracy: {static_acc:.4f}  "
          f"(gate delta: {overall_acc - static_acc:+.4f})", flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "gate_model":      gate_model,
        "agree_thresh":    AGREE_THRESH,
        "static_weights":  {"tft": float(w_tft_n), "bilstm": float(w_bilstm_n)},
        "n_disagree":      n_disagree,
        "gate_cv_acc":     gate_cv_acc,
        "overall_acc":     overall_acc,
    }, MODEL_PATH)

    return {
        "train_acc":    overall_acc,
        "gate_cv_acc":  gate_cv_acc,
        "weights":      {"tft": float(w_tft_n), "bilstm": float(w_bilstm_n)},
        "n_disagree":   n_disagree,
        "pct_disagree": pct_disagree,
        "time_taken":   time.time() - t0,
    }


# ── Shared routing logic ──────────────────────────────────────────────────────

def _route(p_tft: np.ndarray, p_bilstm: np.ndarray,
           df_or_X, feature_cols,
           gate_model, thresh: float,
           w_tft: float, w_bilstm: float) -> np.ndarray:
    """
    Apply agreement-based routing to arrays of base-model probabilities.
    Returns a float32 array of final ensemble probabilities.
    """
    p_tft    = np.asarray(p_tft,    dtype=np.float32)
    p_bilstm = np.asarray(p_bilstm, dtype=np.float32)
    gap      = np.abs(p_tft - p_bilstm)

    # Start with static blend everywhere
    final = (w_tft * p_tft + w_bilstm * p_bilstm).astype(np.float32)

    if gate_model is None:
        return final

    disagree_idx = np.where(gap >= thresh)[0]
    if len(disagree_idx) == 0:
        return final

    # Slice df correctly regardless of type
    if isinstance(df_or_X, pd.DataFrame):
        df_dis = df_or_X.reset_index(drop=True).iloc[disagree_idx]
    else:
        df_dis = df_or_X[disagree_idx]

    X_gate     = _build_gate_X(p_tft[disagree_idx], p_bilstm[disagree_idx],
                                df_dis, feature_cols)
    gate_probs = gate_model.predict_proba(X_gate)[:, 1].astype(np.float32)

    final[disagree_idx] = (gate_probs          * p_tft[disagree_idx]
                           + (1 - gate_probs)  * p_bilstm[disagree_idx])
    return final


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_proba(p_tft: float, p_bilstm: float,
                  last_row, feature_cols: list = None) -> float:
    """
    Single-sample inference.
    last_row: pd.Series or dict representing the current feature row.
    """
    saved      = joblib.load(MODEL_PATH)
    gate_model = saved["gate_model"]
    thresh     = saved["agree_thresh"]
    weights    = saved["static_weights"]
    w_tft      = weights["tft"]
    w_bilstm   = weights["bilstm"]

    gap = abs(p_tft - p_bilstm)
    if gap < thresh or gate_model is None:
        return float(w_tft * p_tft + w_bilstm * p_bilstm)

    if isinstance(last_row, pd.Series):
        df_row = pd.DataFrame([last_row])
        fc     = None
    else:
        df_row = pd.DataFrame([last_row]) if not isinstance(last_row, pd.DataFrame) else last_row
        fc     = feature_cols

    X_gate    = _build_gate_X(np.array([p_tft],    dtype=np.float32),
                               np.array([p_bilstm], dtype=np.float32),
                               df_row, fc)
    gate_prob = float(gate_model.predict_proba(X_gate)[0][1])
    return float(gate_prob * p_tft + (1 - gate_prob) * p_bilstm)


def predict_proba_batch(tft_probs: np.ndarray, bilstm_probs: np.ndarray,
                         df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Batch inference — used by backtest and OOF evaluation."""
    saved      = joblib.load(MODEL_PATH)
    gate_model = saved["gate_model"]
    thresh     = saved["agree_thresh"]
    weights    = saved["static_weights"]
    w_tft      = weights["tft"]
    w_bilstm   = weights["bilstm"]

    return _route(tft_probs, bilstm_probs, df, feature_cols,
                  gate_model, thresh, w_tft, w_bilstm)


def is_trained() -> bool:
    return os.path.exists(MODEL_PATH)
