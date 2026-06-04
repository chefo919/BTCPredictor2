"""
TFT-ACB-XML stacking training pipeline.

Importable library — all steps are exposed as clean functions so Google Colab
(or any orchestrator) can call them directly without duplicating logic.

Entry points
------------
run_full_training_pipeline(config_override)   ← primary API
load_and_label_data()                         ← step 0
train_tft_ensemble()                          ← phase 1
train_bilstm_ensemble()                       ← phase 2
train_meta_learner()                          ← phase 3

CLI (unchanged behaviour):
    python training/train.py --force [--test] [--config]
"""

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["DNNL_VERBOSE"]          = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
try:
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
except RuntimeError:
    pass  # already initialized (e.g. imported after another TF module)

from features.engineer import (BILSTM_FEAT_COLS, TFT_FEAT_COLS, get_feature_groups)
from models import tft_model, bilstm_model, meta_model

ROOT       = os.path.dirname(os.path.dirname(__file__))
YEARLY_DIR = os.path.join(ROOT, "data", "yearly_merged")
STAMP_PATH = os.path.join(ROOT, "models", "saved", "last_train_rows.txt")
ACC_PATH   = os.path.join(ROOT, "models", "saved", "model_accuracies.json")

from config import (TRAINING_CUTOFF_DATE, LABEL_TP_PCT, LABEL_SL_PCT,
                    HORIZON_TFT, HORIZON_BILSTM,
                    PURGE_ROWS_TFT, PURGE_ROWS_BILSTM)
from training.labels import apply_triple_barrier

TRAINING_START = "2022-02-01"  # skip first 30d — rolling window warm-up period

# Prefer yearly splits (pushed to GitHub); fall back to monolithic file if present
_YEARLY_1M_DIR = os.path.join(ROOT, "data", "yearly_1m")
_MONO_1M       = os.path.join(ROOT, "data", "btc_1m.csv")
PATH_1M = _YEARLY_1M_DIR if os.path.isdir(_YEARLY_1M_DIR) else _MONO_1M

# Feature groups — loaded once at module level (deterministic, no I/O)
_GROUPS           = get_feature_groups()
BILSTM_FEATURES   = _GROUPS["bilstm"]        # m15_* + h1_*  (24 features)
TFT_DYN_FEATURES  = _GROUPS["tft_dynamic"]   # h4_* + d1_*   (24 features)
TFT_STA_FEATURES  = _GROUPS["tft_static"]    # [] — no static covariates
ALL_GATE_FEATURES = TFT_DYN_FEATURES + BILSTM_FEATURES


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _load_saved_acc() -> dict:
    if os.path.exists(ACC_PATH):
        with open(ACC_PATH) as f:
            return json.load(f)
    return {}


def get_last_train_rows() -> int:
    if os.path.exists(STAMP_PATH):
        with open(STAMP_PATH) as f:
            return int(f.read().strip())
    return 0


def save_train_rows(n: int):
    os.makedirs(os.path.dirname(STAMP_PATH), exist_ok=True)
    with open(STAMP_PATH, "w") as f:
        f.write(str(n))


def should_retrain(current_rows: int, threshold: int = 1000) -> bool:
    last = get_last_train_rows()
    return (current_rows - last) >= threshold or last == 0


def _load_yearly(suffix: str) -> pd.DataFrame:
    """Load all yearly CSV files matching suffix into one sorted DataFrame."""
    if not os.path.exists(YEARLY_DIR):
        print(f"data/yearly_merged/ not found. Run features/engineer.py first.", flush=True)
        sys.exit(1)
    start_ts  = pd.Timestamp(TRAINING_START,        tz="UTC")
    cutoff_ts = pd.Timestamp(TRAINING_CUTOFF_DATE,  tz="UTC")
    dfs = []
    for fname in sorted(os.listdir(YEARLY_DIR)):
        if not fname.endswith(suffix):
            continue
        df_y = pd.read_csv(os.path.join(YEARLY_DIR, fname), parse_dates=["time"])
        if df_y["time"].dt.tz is None:
            df_y["time"] = pd.to_datetime(df_y["time"], utc=True)
        df_y = df_y[(df_y["time"] >= start_ts) & (df_y["time"] <= cutoff_ts)]
        if not df_y.empty:
            dfs.append(df_y)
    if not dfs:
        print(f"No data in {suffix} files between {TRAINING_START} and {TRAINING_CUTOFF_DATE}.",
              flush=True)
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)


def load_bilstm_data() -> pd.DataFrame:
    """Load 15min-sampled bilstm_merged files (m15_* + h1_* + close + time)."""
    df = _load_yearly("_bilstm_merged.csv")
    print(f"BiLSTM data: {len(df):,} rows "
          f"({TRAINING_START} -> {TRAINING_CUTOFF_DATE}, {len(BILSTM_FEAT_COLS)} features).",
          flush=True)
    return df


def load_tft_data() -> pd.DataFrame:
    """Load 4h-sampled tft_merged files (h4_* + d1_* + close + time)."""
    df = _load_yearly("_tft_merged.csv")
    print(f"TFT data: {len(df):,} rows "
          f"({TRAINING_START} -> {TRAINING_CUTOFF_DATE}, {len(TFT_FEAT_COLS)} features).",
          flush=True)
    return df


# ── Modular pipeline steps ─────────────────────────────────────────────────────

def load_and_label_data(test_mode: bool = False) -> tuple:
    """
    Step 0: Load both datasets and apply triple-barrier labels.

    Returns (df_bilstm, df_tft) — each DataFrame has a 'target' column added
    and NaN-label rows dropped.
    """
    df_bilstm = load_bilstm_data()
    df_tft    = load_tft_data()

    if test_mode:
        df_bilstm = df_bilstm.tail(5000).copy()
        df_tft    = df_tft.tail(1000).copy()
        print(f"[TEST MODE] BiLSTM: {len(df_bilstm):,} rows  TFT: {len(df_tft):,} rows",
              flush=True)

    print("[0/3] Applying triple-barrier labels...", flush=True)
    df_tft    = apply_triple_barrier(df_tft,    PATH_1M, LABEL_TP_PCT, LABEL_SL_PCT, HORIZON_TFT)
    df_bilstm = apply_triple_barrier(df_bilstm, PATH_1M, LABEL_TP_PCT, LABEL_SL_PCT, HORIZON_BILSTM)

    return df_bilstm, df_tft


def train_tft_ensemble(df_tft: pd.DataFrame, force: bool = False) -> tuple:
    """
    Phase 1: Train TFT ensemble (3 seeds with diverse hyperparameters).

    Skips if models are already saved and force=False.
    Returns (tft_acc, tft_val_err) as floats.
    """
    saved_acc = _load_saved_acc()

    if tft_model.is_trained() and not force:
        tft_acc     = saved_acc.get("tft",     0.51)
        tft_val_err = 1.0 - saved_acc.get("tft_val", tft_acc)
        print(f"[1/3] TFT already trained — skipping  (saved acc: {tft_acc:.3f})")
        return float(tft_acc), float(tft_val_err)

    print(f"[1/3] Training TFT ensemble ({tft_model.N_ENSEMBLE} seeds, "
          f"SEQ_LEN={tft_model.SEQ_LEN}, HORIZON={tft_model.HORIZON}min, "
          f"batch={tft_model.BATCH})...", flush=True)
    t0          = time.time()
    results     = tft_model.train_ensemble(df_tft, TFT_DYN_FEATURES, TFT_STA_FEATURES)
    elapsed     = time.time() - t0
    tft_acc     = float(np.mean([r["test_acc"] for r in results]))
    tft_val_err = 1.0 - float(np.mean([r.get("val_acc", r["test_acc"]) for r in results]))
    print(f"\nTFT ensemble complete | Avg accuracy: {tft_acc:.3f} | Time: {_fmt(elapsed)}",
          flush=True)
    return tft_acc, tft_val_err


def train_bilstm_ensemble(df_bilstm: pd.DataFrame, force: bool = False) -> tuple:
    """
    Phase 2: Train BiLSTM/ACB ensemble (3 seeds with diverse hyperparameters).

    Skips if models are already saved and force=False.
    Returns (bilstm_acc, bilstm_val_err) as floats.
    """
    saved_acc = _load_saved_acc()

    if bilstm_model.is_trained() and not force:
        bilstm_acc     = saved_acc.get("bilstm",     0.51)
        bilstm_val_err = 1.0 - saved_acc.get("bilstm_val", bilstm_acc)
        print(f"[2/3] BiLSTM already trained — skipping  (saved acc: {bilstm_acc:.3f})")
        return float(bilstm_acc), float(bilstm_val_err)

    print(f"[2/3] Training BiLSTM ensemble ({bilstm_model.N_ENSEMBLE} seeds, "
          f"SEQ_LEN={bilstm_model.SEQ_LEN}, HORIZON={bilstm_model.HORIZON}min, "
          f"batch={bilstm_model.BATCH})...", flush=True)
    t0             = time.time()
    results        = bilstm_model.train_ensemble(df_bilstm, BILSTM_FEATURES)
    elapsed        = time.time() - t0
    bilstm_acc     = float(np.mean([r["test_acc"] for r in results]))
    bilstm_val_err = 1.0 - float(np.mean([r.get("val_acc", r["test_acc"])
                                           for r in results]))
    print(f"\nBiLSTM ensemble complete | Avg accuracy: {bilstm_acc:.3f} | "
          f"Time: {_fmt(elapsed)}", flush=True)
    return bilstm_acc, bilstm_val_err


def train_meta_learner(df_tft: pd.DataFrame, df_bilstm: pd.DataFrame,
                       tft_val_err: float, bilstm_val_err: float) -> dict:
    """
    Phase 3: Generate OOF predictions and train the XGBoost agreement-based routing gate.

    OOF split layout — 72h purge matches MAX_HOLD_MIN = 4320 min = 3 days:
      TFT   (4h-sampled):   [0–70% train] [+18 row purge = 72h] [70–90% OOF] [test]
      BiLSTM (15m-sampled): [0–70% train] [+288 row purge = 72h] [70–90% OOF] [test]

    BiLSTM OOF is aligned onto TFT timestamps via merge_asof with 15-min tolerance.
    Tighter tolerance (was 3h) preserves the micro-momentum context the gate relies on.

    Returns meta_results dict from meta_model.train().
    """
    print("[3/3] Generating OOF predictions and training XGBoost meta-learner...",
          flush=True)

    df_tft_clean    = df_tft.dropna(subset=TFT_DYN_FEATURES).reset_index(drop=True)
    df_bilstm_clean = df_bilstm.dropna(subset=BILSTM_FEATURES).reset_index(drop=True)
    n_tft    = len(df_tft_clean)
    n_bilstm = len(df_bilstm_clean)

    # 72h purge — PURGE_ROWS_TFT=18 (4h rows), PURGE_ROWS_BILSTM=288 (15min rows)
    tft_train_end = int(n_tft    * 0.70)
    tft_oof_start = tft_train_end  + PURGE_ROWS_TFT
    tft_oof_end   = int(n_tft    * 0.90)

    bi_train_end  = int(n_bilstm * 0.70)
    bi_oof_start  = bi_train_end   + PURGE_ROWS_BILSTM
    bi_oof_end    = int(n_bilstm * 0.90)

    df_tft_oof    = df_tft_clean.iloc[tft_oof_start:tft_oof_end].copy()
    df_bilstm_oof = df_bilstm_clean.iloc[bi_oof_start:bi_oof_end].copy()

    print(f"  TFT OOF:    rows {tft_oof_start:,}–{tft_oof_end:,}  "
          f"({len(df_tft_oof):,} rows, purge={PURGE_ROWS_TFT} rows = 72h)",  flush=True)
    print(f"  BiLSTM OOF: rows {bi_oof_start:,}–{bi_oof_end:,}  "
          f"({len(df_bilstm_oof):,} rows, purge={PURGE_ROWS_BILSTM} rows = 72h)", flush=True)

    val_X_dyn = df_tft_oof[TFT_DYN_FEATURES].values.astype("float32")
    val_X_sta = (df_tft_oof[TFT_STA_FEATURES].values.astype("float32")
                 if TFT_STA_FEATURES
                 else np.zeros((len(df_tft_oof), 0), dtype="float32"))
    val_X_bi  = df_bilstm_oof[BILSTM_FEATURES].values.astype("float32")

    print(f"  Generating TFT OOF predictions on {len(df_tft_oof):,} rows...", flush=True)
    val_tft_probs = tft_model.predict_proba_batch(val_X_dyn, val_X_sta,
                                                   timestamps=df_tft_oof["time"])

    print(f"  Generating BiLSTM OOF predictions on {len(df_bilstm_oof):,} rows...", flush=True)
    val_bilstm_probs_full = bilstm_model.predict_proba_batch(val_X_bi)

    # ── Align BiLSTM OOF onto TFT timestamps ─────────────────────────────────
    # tolerance=15min: each 4h TFT row maps to the nearest BiLSTM row within ±15min.
    # BiLSTM data is at 15min resolution so there is always a match within tolerance.
    # Tighter than the former 3h to avoid carrying stale micro-momentum context.
    df_bi_oof_probs            = df_bilstm_oof[["time"]].copy()
    df_bi_oof_probs["bilstm_prob"] = val_bilstm_probs_full
    df_tft_oof_probs           = df_tft_oof[["time"]].copy()
    df_tft_oof_probs["tft_idx"] = np.arange(len(df_tft_oof))

    df_bi_oof_probs  = df_bi_oof_probs.sort_values("time")
    df_tft_oof_probs = df_tft_oof_probs.sort_values("time")

    aligned = pd.merge_asof(
        df_tft_oof_probs,
        df_bi_oof_probs,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
    )
    aligned["bilstm_prob"] = aligned["bilstm_prob"].fillna(0.5)

    val_bilstm_probs_aligned = aligned["bilstm_prob"].values.astype("float32")
    val_tft_probs_aligned    = val_tft_probs[aligned["tft_idx"].values]

    # ── Build combined gate snapshot: TFT features + nearest BiLSTM features ──
    bilstm_oof_sorted = df_bilstm_oof.sort_values("time").reset_index(drop=True)
    df_combined = pd.merge_asof(
        df_tft_oof.sort_values("time").reset_index(drop=True),
        bilstm_oof_sorted[["time"] + BILSTM_FEATURES],
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
    )
    for col in BILSTM_FEATURES:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].fillna(0.0)

    t_meta       = time.time()
    meta_results = meta_model.train(df_combined,
                                    val_tft_probs_aligned,
                                    val_bilstm_probs_aligned,
                                    tft_val_err, bilstm_val_err,
                                    ALL_GATE_FEATURES)
    print(f"\nMeta-learner complete | CV acc: {meta_results['train_acc']:.3f} | "
          f"Time: {_fmt(time.time() - t_meta)}", flush=True)
    return meta_results


# ── Master orchestration ───────────────────────────────────────────────────────

def run_full_training_pipeline(config_override: dict = None) -> dict:
    """
    Master orchestration — runs all three training phases and persists results.

    config_override keys (all optional)
    ------------------------------------
    batch_tft:       int   TFT batch size          (default: BATCH_TFT from config; use 512 on GPU)
    batch_bilstm:    int   BiLSTM batch size        (default: BATCH_BILSTM from config)
    n_epochs_tft:    int   TFT max epochs override
    n_epochs_bilstm: int   BiLSTM max epochs override
    force:           bool  Retrain even if models already exist (default: False)
    test_mode:       bool  Use small data subset for a fast smoke-test (default: False)

    Returns
    -------
    dict with keys: tft_acc, bilstm_acc, meta_acc, gate_cv_acc, weights
    """
    override  = config_override or {}
    force     = bool(override.get("force",     False))
    test_mode = bool(override.get("test_mode", False))

    # Apply batch / epoch overrides to model modules before any training starts
    if "batch_tft"       in override: tft_model.BATCH       = int(override["batch_tft"])
    if "batch_bilstm"    in override: bilstm_model.BATCH    = int(override["batch_bilstm"])
    if "n_epochs_tft"    in override: tft_model.N_EPOCHS    = int(override["n_epochs_tft"])
    if "n_epochs_bilstm" in override: bilstm_model.N_EPOCHS = int(override["n_epochs_bilstm"])
    if test_mode:
        tft_model.N_EPOCHS    = 3
        bilstm_model.N_EPOCHS = 3

    SEP = "=" * 50
    t0  = time.time()
    print()
    print(SEP)
    print("TFT-ACB-XML STACKING TRAINING  (arXiv 2602.12380)")
    print(SEP)
    print(f"Phase 1/3: TFT    macro   SEQ={tft_model.SEQ_LEN} × 4h = 30 days  "
          f"| dynamic={len(TFT_DYN_FEATURES)}  | HORIZON={tft_model.HORIZON}min")
    print(f"Phase 2/3: BiLSTM micro   SEQ={bilstm_model.SEQ_LEN} × 15min = 48h  "
          f"| features={len(BILSTM_FEATURES)}  | HORIZON={bilstm_model.HORIZON}min")
    print(f"Phase 3/3: XGBoost meta   purge=72h ({PURGE_ROWS_TFT} TFT rows / "
          f"{PURGE_ROWS_BILSTM} BiLSTM rows)  | merge_tol=15min")
    print(f"Batch sizes: TFT={tft_model.BATCH}  BiLSTM={bilstm_model.BATCH}  "
          f"force={force}  test_mode={test_mode}")
    print(SEP)
    print()

    # ── Phases 0 → 3 ─────────────────────────────────────────────────────────
    df_bilstm, df_tft = load_and_label_data(test_mode=test_mode)
    current_rows = len(df_bilstm) + len(df_tft)

    tft_acc,    tft_val_err    = train_tft_ensemble(df_tft,    force=force)
    print()
    bilstm_acc, bilstm_val_err = train_bilstm_ensemble(df_bilstm, force=force)
    print()
    meta_results               = train_meta_learner(df_tft, df_bilstm,
                                                     tft_val_err, bilstm_val_err)

    save_train_rows(current_rows)

    # ── Persist accuracy summary ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(ACC_PATH), exist_ok=True)
    with open(ACC_PATH, "w") as f:
        json.dump({
            "tft":        tft_acc,
            "tft_val":    1.0 - tft_val_err,
            "bilstm":     bilstm_acc,
            "bilstm_val": 1.0 - bilstm_val_err,
            "meta":       meta_results["train_acc"],
        }, f, indent=2)

    cutoff_path = os.path.join(ROOT, "models", "saved", "training_cutoff.txt")
    with open(cutoff_path, "w") as f:
        f.write(str(df_tft["time"].iloc[-1]))

    total_elapsed = time.time() - t0
    w = meta_results.get("weights", {})
    print()
    print(SEP)
    print("TFT-ACB-XML Training complete!")
    print(f"[Training] TFT accuracy:      {tft_acc:.3f}  "
          f"(ensemble of {tft_model.N_ENSEMBLE} seeds)")
    print(f"[Training] BiLSTM accuracy:   {bilstm_acc:.3f}  "
          f"(ensemble of {bilstm_model.N_ENSEMBLE} seeds)")
    print(f"[Training] Meta routing acc:  {meta_results['train_acc']:.3f}  "
          f"(OOF, agreement-based gate)")
    print(f"[Training] Gate CV acc:       {meta_results.get('gate_cv_acc', 0):.3f}  "
          f"(on {meta_results.get('n_disagree', 0):,} disagreement rows, "
          f"{meta_results.get('pct_disagree', 0):.1f}% of OOF)")
    print(f"[Training] Static fallback — TFT: {w.get('tft',0):.3f}  "
          f"BiLSTM: {w.get('bilstm',0):.3f}")
    print(f"[Training] Data window: {TRAINING_START} -> {TRAINING_CUTOFF_DATE}")
    print(f"Models saved to:  models/saved/")
    print(f"Total time:       {_fmt(total_elapsed)}")
    print(SEP)

    return {
        "tft_acc":     tft_acc,
        "bilstm_acc":  bilstm_acc,
        "meta_acc":    meta_results["train_acc"],
        "gate_cv_acc": meta_results.get("gate_cv_acc", 0.5),
        "weights":     w,
    }


# ── Legacy CLI wrapper (backward-compatible) ──────────────────────────────────

def run_training(force: bool = False, test_mode: bool = False):
    """CLI guard + delegate. Use --force flag from terminal to proceed."""
    if not force:
        print("Use --force to run training. Skipping.")
        return
    run_full_training_pipeline(config_override={"test_mode": test_mode})


def _ram_available() -> str:
    try:
        import psutil
        return f"{psutil.virtual_memory().available / 1024**3:.1f} GB"
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemAvailable" in line:
                    return f"{int(line.split()[1]) / 1024**2:.1f} GB"
    except Exception:
        pass
    return "unknown"


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _force       = "--force"  in sys.argv
    _test_mode   = "--test"   in sys.argv
    _config_only = "--config" in sys.argv

    if _config_only:
        _df_tft    = load_tft_data()
        _df_bilstm = load_bilstm_data()
        print(f"TFT config:    {len(_df_tft):,} rows, {len(TFT_FEAT_COLS)} features, "
              f"HORIZON={tft_model.HORIZON}")
        print(f"BiLSTM config: {len(_df_bilstm):,} rows, {len(BILSTM_FEAT_COLS)} features, "
              f"HORIZON={bilstm_model.HORIZON}")
    else:
        run_training(force=_force, test_mode=_test_mode)
