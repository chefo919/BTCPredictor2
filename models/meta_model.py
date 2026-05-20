"""
XGBoost Meta-Learner with Error-Reciprocal Weighting for TFT-ACB-XML stacking.

Corresponds to the XML (XGBoost Meta-Learner) component in:
  "TFT-ACB-XML: Decision-Level Integration of Customized TFT and Attention-BiLSTM
   with XGBoost Meta-Learner for BTC Price Forecasting" — Din & Khan (2026), arXiv 2602.12380.

Error-Reciprocal Weighting (paper's key innovation):
  w_tft    = 1 / val_error_tft    (lower error → higher trust)
  w_bilstm = 1 / val_error_bilstm
  Normalized weights applied to base model probabilities before feeding XGBoost.
  XGBoost then captures non-linear residuals the base models miss.

Meta-features: [weighted_p_tft, weighted_p_bilstm, raw_p_tft, raw_p_bilstm,
                 d1_rsi, h4_rsi, h1_rsi, d1_ema9_ratio, d1_ema50_ratio]
"""
import os, sys
import numpy as np
import joblib
import pandas as pd

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import HORIZON_META as HORIZON

MODEL_DIR  = os.path.join(ROOT, "models", "saved")
MODEL_PATH = os.path.join(MODEL_DIR, "meta_xgb.pkl")

META_RAW_FEATURES = [
    "d1_rsi", "h4_rsi", "h1_rsi", "m30_rsi",
    "d1_ema9_ratio", "d1_ema50_ratio",
    "h4_atr_norm",       # volatility regime — high = trending, trust signals more
    "h1_vol_ratio",      # volume conviction
    "h4_bb_width",       # band width = volatility proxy
    "d1_macd_diff_pct",  # macro momentum direction
]


def _get_col(df_or_X, name, feature_cols):
    """Extract a named feature column from dataframe or numpy array."""
    if isinstance(df_or_X, pd.DataFrame):
        return df_or_X[name].values if name in df_or_X.columns else np.zeros(len(df_or_X))
    fc = list(feature_cols) if feature_cols else []
    if name in fc:
        return df_or_X[:, fc.index(name)]
    return np.zeros(len(df_or_X))


def _build_meta_X(p_tft_w, p_bilstm_w, p_tft_raw, p_bilstm_raw,
                   df_or_X, feature_cols=None) -> np.ndarray:
    """Build meta-feature matrix: weighted probs + raw probs + key technical features."""
    n = len(p_tft_raw)
    cols = [
        np.array(p_tft_w).reshape(n),       # error-weighted TFT prob
        np.array(p_bilstm_w).reshape(n),     # error-weighted BiLSTM prob
        np.array(p_tft_raw).reshape(n),      # raw TFT prob
        np.array(p_bilstm_raw).reshape(n),   # raw BiLSTM prob
    ]
    for feat in META_RAW_FEATURES:
        cols.append(_get_col(df_or_X, feat, feature_cols))
    return np.column_stack(cols).astype(np.float32)


def train(val_df: pd.DataFrame,
          val_tft_probs: np.ndarray,
          val_bilstm_probs: np.ndarray,
          tft_val_error: float,
          bilstm_val_error: float,
          feature_cols: list) -> dict:
    """
    Train XGBoost meta-learner on out-of-fold predictions from the 15% validation slice.

    val_tft_probs, val_bilstm_probs: predictions from base models on val_df
    tft_val_error, bilstm_val_error: 1 - val_acc for each base model
    """
    import time
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    t0 = time.time()

    # Error-reciprocal weights (paper's key innovation)
    w_tft    = 1.0 / max(float(tft_val_error),    1e-6)
    w_bilstm = 1.0 / max(float(bilstm_val_error), 1e-6)
    w_total  = w_tft + w_bilstm
    w_tft_n, w_bilstm_n = w_tft / w_total, w_bilstm / w_total

    print(f"  Error-reciprocal weights — TFT: {w_tft_n:.3f}  BiLSTM: {w_bilstm_n:.3f}",
          flush=True)

    p_tft_w    = val_tft_probs    * w_tft_n
    p_bilstm_w = val_bilstm_probs * w_bilstm_n

    # Build target for meta-learner (same binary target)
    val_df = val_df.copy()
    val_df["target_meta"] = (val_df["close"].shift(-HORIZON) > val_df["close"]).astype(int)
    val_df = val_df.dropna(subset=["target_meta"])

    # Align arrays to valid rows (HORIZON rows lost from shift)
    n_valid = len(val_df)
    p_tft_raw_v    = val_tft_probs[-n_valid:]
    p_bilstm_raw_v = val_bilstm_probs[-n_valid:]
    p_tft_w_v      = p_tft_w[-n_valid:]
    p_bilstm_w_v   = p_bilstm_w[-n_valid:]

    X_meta = _build_meta_X(p_tft_w_v, p_bilstm_w_v, p_tft_raw_v, p_bilstm_raw_v,
                            val_df, feature_cols)
    y_meta = val_df["target_meta"].values

    print(f"  Meta-learner training: {len(y_meta):,} rows  "
          f"label balance: {y_meta.mean():.3f}", flush=True)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0,
    )
    model.fit(X_meta, y_meta)
    preds = model.predict(X_meta)
    train_acc = float(accuracy_score(y_meta, preds))
    print(f"  Meta-learner train accuracy: {train_acc:.4f}", flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model":   model,
        "weights": {"tft": float(w_tft_n), "bilstm": float(w_bilstm_n)},
        "tft_val_error":    float(tft_val_error),
        "bilstm_val_error": float(bilstm_val_error),
    }, MODEL_PATH)

    return {"train_acc": train_acc, "weights": {"tft": w_tft_n, "bilstm": w_bilstm_n},
            "time_taken": time.time() - t0}


def predict_proba(p_tft: float, p_bilstm: float,
                  last_row, feature_cols: list = None) -> float:
    """
    Generate stacked prediction for a single sample.
    last_row: pd.Series or 1D array of the current feature row.
    """
    saved    = joblib.load(MODEL_PATH)
    model    = saved["model"]
    weights  = saved["weights"]
    w_tft, w_bilstm = weights["tft"], weights["bilstm"]

    p_tft_w    = p_tft    * w_tft
    p_bilstm_w = p_bilstm * w_bilstm

    df_row = pd.DataFrame([last_row], columns=feature_cols) if feature_cols else pd.DataFrame()
    X_meta = _build_meta_X([p_tft_w], [p_bilstm_w], [p_tft], [p_bilstm],
                            df_row, feature_cols)
    return float(model.predict_proba(X_meta)[0][1])


def predict_proba_batch(tft_probs: np.ndarray, bilstm_probs: np.ndarray,
                         df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Batch inference for backtest — parallel to row-wise predict_proba."""
    saved   = joblib.load(MODEL_PATH)
    model   = saved["model"]
    weights = saved["weights"]
    w_tft, w_bilstm = weights["tft"], weights["bilstm"]

    p_tft_w    = tft_probs    * w_tft
    p_bilstm_w = bilstm_probs * w_bilstm

    X_meta = _build_meta_X(p_tft_w, p_bilstm_w, tft_probs, bilstm_probs,
                            df, feature_cols)
    return model.predict_proba(X_meta)[:, 1].astype(np.float32)


def is_trained() -> bool:
    return os.path.exists(MODEL_PATH)
