from sklearnex import patch_sklearn
patch_sklearn()

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["DNNL_VERBOSE"]          = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import sys
import time
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

from config import TRAINING_CUTOFF_DATE
TRAINING_START = "2022-02-01"  # skip first 30d — rolling window warm-up period


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


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
    """Load all yearly CSV files matching the given suffix into a single DataFrame."""
    if not os.path.exists(YEARLY_DIR):
        print(f"data/yearly_merged/ not found. Run features/engineer.py first.", flush=True)
        sys.exit(1)
    start_ts  = pd.Timestamp(TRAINING_START, tz="UTC")
    cutoff_ts = pd.Timestamp(TRAINING_CUTOFF_DATE, tz="UTC")
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
    df = pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
    return df


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
                    kb = int(line.split()[1])
                    return f"{kb / 1024**2:.1f} GB"
    except Exception:
        pass
    return "unknown"


def run_training(force: bool = False, test_mode: bool = False):
    groups           = get_feature_groups()
    BILSTM_FEATURES  = groups["bilstm"]        # m15/h1 — 24 features
    TFT_DYN_FEATURES = groups["tft_dynamic"]   # h4/d1  — 24 features
    TFT_STA_FEATURES = groups["tft_static"]    # none

    df_bilstm = load_bilstm_data()
    df_tft    = load_tft_data()

    current_rows = len(df_bilstm) + len(df_tft)

    if not force:
        print("Use --force to run training. Skipping.")
        return

    if test_mode:
        df_bilstm = df_bilstm.tail(5000).copy()
        df_tft    = df_tft.tail(1000).copy()
        print(f"[TEST MODE] BiLSTM: {len(df_bilstm):,} rows, TFT: {len(df_tft):,} rows, 3 epochs.",
              flush=True)

    t0  = time.time()
    SEP = "=" * 46

    print()
    print(SEP)
    print("TFT-ACB-XML STACKING TRAINING (arXiv 2602.12380)")
    print(SEP)
    print(f"Phase 1/3: TFT  (macro, SEQ_LEN={tft_model.SEQ_LEN}, 4h-sampled, "
          f"dynamic={len(TFT_DYN_FEATURES)}, horizon={tft_model.HORIZON}min)")
    print(f"Phase 2/3: BiLSTM/ACB (micro, SEQ_LEN={bilstm_model.SEQ_LEN}, 15min-sampled, "
          f"features={len(BILSTM_FEATURES)}, horizon={bilstm_model.HORIZON}min)")
    print(f"Phase 3/3: XGBoost meta-learner (full multi-timeframe gate snapshot)")
    print(f"TFT rows:    {len(df_tft):,}  |  BiLSTM rows: {len(df_bilstm):,}")
    print(f"Target:      price direction {tft_model.HORIZON} minutes (1 day) ahead")
    print(SEP)
    print()

    import json as _json_acc
    _acc_path = os.path.join(ROOT, "models", "saved", "model_accuracies.json")
    _saved_acc = {}
    if os.path.exists(_acc_path):
        with open(_acc_path) as _f:
            _saved_acc = _json_acc.load(_f)

    if test_mode:
        tft_model.N_EPOCHS    = 3
        bilstm_model.N_EPOCHS = 3

    # ── Phase 1: Train TFT on 4h-sampled data ────────────────────────────────
    if tft_model.is_trained():
        tft_acc     = _saved_acc.get("tft", 0.51)
        tft_val_err = 1.0 - _saved_acc.get("tft_val", tft_acc)
        print(f"[1/3] TFT already trained — skipping  (saved acc: {tft_acc:.3f})")
    else:
        print(f"[1/3] Training TFT ensemble ({tft_model.N_ENSEMBLE} seeds, "
              f"SEQ_LEN={tft_model.SEQ_LEN}, HORIZON={tft_model.HORIZON})...")
        t_tft        = time.time()
        tft_results  = tft_model.train_ensemble(df_tft, TFT_DYN_FEATURES, TFT_STA_FEATURES)
        tft_elapsed  = time.time() - t_tft
        tft_acc      = float(np.mean([r["test_acc"] for r in tft_results]))
        tft_val_err  = 1.0 - float(np.mean([r.get("val_acc", r["test_acc"]) for r in tft_results]))
        print(f"\nTFT ensemble complete | Avg accuracy: {tft_acc:.3f} | Time: {_fmt(tft_elapsed)}")
    print()

    # ── Phase 2: Train BiLSTM/ACB on 15min-sampled data ──────────────────────
    if bilstm_model.is_trained():
        bilstm_acc     = _saved_acc.get("bilstm", 0.51)
        bilstm_val_err = 1.0 - _saved_acc.get("bilstm_val", bilstm_acc)
        print(f"[2/3] BiLSTM already trained — skipping  (saved acc: {bilstm_acc:.3f})")
    else:
        print(f"[2/3] Training BiLSTM ensemble ({bilstm_model.N_ENSEMBLE} seeds, "
              f"SEQ_LEN={bilstm_model.SEQ_LEN}, HORIZON={bilstm_model.HORIZON})...")
        t_bilstm        = time.time()
        bilstm_results  = bilstm_model.train_ensemble(df_bilstm, BILSTM_FEATURES)
        bilstm_elapsed  = time.time() - t_bilstm
        bilstm_acc      = float(np.mean([r["test_acc"] for r in bilstm_results]))
        bilstm_val_err  = 1.0 - float(np.mean([r.get("val_acc", r["test_acc"])
                                                for r in bilstm_results]))
        print(f"\nBiLSTM ensemble complete | Avg accuracy: {bilstm_acc:.3f} | "
              f"Time: {_fmt(bilstm_elapsed)}")
    print()

    # ── Phase 3: OOF predictions → train meta ────────────────────────────────
    # TFT timeline (4h rows):    [0-70% train] [+6 row gap] [72-90% OOF] [+6 row gap] [test]
    # BiLSTM timeline (15m rows): [0-70% train] [+96 row gap] [72-90% OOF] [+96 row gap] [test]
    # 6 TFT rows = 24h; 96 BiLSTM rows = 24h — same real-time purge gap.
    # OOF gate snapshot combines both CSVs via nearest-time join.
    print("[3/3] Generating OOF predictions and training XGBoost meta-learner...")

    df_tft_clean    = df_tft.dropna(subset=TFT_DYN_FEATURES).reset_index(drop=True)
    df_bilstm_clean = df_bilstm.dropna(subset=BILSTM_FEATURES).reset_index(drop=True)
    n_tft    = len(df_tft_clean)
    n_bilstm = len(df_bilstm_clean)

    PURGE_TFT    = 6    # 24h / 4h = 6 rows in tft_merged
    PURGE_BILSTM = 96   # 24h / 15min = 96 rows in bilstm_merged

    tft_train_end  = int(n_tft    * 0.70)
    tft_oof_start  = tft_train_end  + PURGE_TFT
    tft_oof_end    = int(n_tft    * 0.90)

    bi_train_end   = int(n_bilstm * 0.70)
    bi_oof_start   = bi_train_end   + PURGE_BILSTM
    bi_oof_end     = int(n_bilstm * 0.90)

    df_tft_oof    = df_tft_clean.iloc[tft_oof_start:tft_oof_end].copy()
    df_bilstm_oof = df_bilstm_clean.iloc[bi_oof_start:bi_oof_end].copy()

    print(f"  TFT OOF:    rows {tft_oof_start:,}-{tft_oof_end:,} ({len(df_tft_oof):,} rows)",
          flush=True)
    print(f"  BiLSTM OOF: rows {bi_oof_start:,}-{bi_oof_end:,} ({len(df_bilstm_oof):,} rows)",
          flush=True)

    val_X_dyn = df_tft_oof[TFT_DYN_FEATURES].values.astype("float32")
    val_X_sta = df_tft_oof[TFT_STA_FEATURES].values.astype("float32")
    val_X_bi  = df_bilstm_oof[BILSTM_FEATURES].values.astype("float32")

    print(f"  Generating TFT OOF predictions on {len(df_tft_oof):,} rows...", flush=True)
    val_tft_probs = tft_model.predict_proba_batch(val_X_dyn, val_X_sta)

    print(f"  Generating BiLSTM OOF predictions on {len(df_bilstm_oof):,} rows...", flush=True)
    val_bilstm_probs_full = bilstm_model.predict_proba_batch(val_X_bi)

    # ── Align BiLSTM OOF onto TFT timestamps for the combined gate snapshot ───
    # Each TFT row (4h) maps to the nearest BiLSTM row (15min) via merge_asof.
    # This gives the gate a full multi-timeframe snapshot at each 4h decision point.
    df_bi_oof_probs = df_bilstm_oof[["time"]].copy()
    df_bi_oof_probs["bilstm_prob"] = val_bilstm_probs_full

    df_tft_oof_probs = df_tft_oof[["time"]].copy()
    df_tft_oof_probs["tft_idx"] = np.arange(len(df_tft_oof))

    # Sort required for merge_asof
    df_bi_oof_probs  = df_bi_oof_probs.sort_values("time")
    df_tft_oof_probs = df_tft_oof_probs.sort_values("time")

    aligned = pd.merge_asof(
        df_tft_oof_probs,
        df_bi_oof_probs,
        on="time",
        direction="nearest",
    )

    val_bilstm_probs_aligned = aligned["bilstm_prob"].values.astype("float32")
    val_tft_probs_aligned    = val_tft_probs[aligned["tft_idx"].values]

    # Build combined snapshot DataFrame: TFT features + nearest BiLSTM features
    bilstm_oof_sorted = df_bilstm_oof.sort_values("time").reset_index(drop=True)
    df_combined = pd.merge_asof(
        df_tft_oof.sort_values("time").reset_index(drop=True),
        bilstm_oof_sorted[["time"] + BILSTM_FEATURES],
        on="time",
        direction="nearest",
    )

    ALL_GATE_FEATURES = TFT_DYN_FEATURES + BILSTM_FEATURES

    t_meta       = time.time()
    meta_results = meta_model.train(df_combined,
                                    val_tft_probs_aligned,
                                    val_bilstm_probs_aligned,
                                    tft_val_err, bilstm_val_err,
                                    ALL_GATE_FEATURES)
    meta_elapsed = time.time() - t_meta
    print(f"\nMeta-learner complete | CV acc: {meta_results['train_acc']:.3f} | "
          f"Time: {_fmt(meta_elapsed)}")

    save_train_rows(current_rows)

    # ── Save accuracy and cutoff ──────────────────────────────────────────────
    import json as _json
    acc_path = os.path.join(ROOT, "models", "saved", "model_accuracies.json")
    with open(acc_path, "w") as _f:
        _json.dump({"tft": tft_acc, "tft_val": 1.0 - tft_val_err,
                    "bilstm": bilstm_acc, "bilstm_val": 1.0 - bilstm_val_err,
                    "meta": meta_results["train_acc"]}, _f, indent=2)

    cutoff_ts   = str(df_tft["time"].iloc[-1])
    cutoff_path = os.path.join(ROOT, "models", "saved", "training_cutoff.txt")
    with open(cutoff_path, "w") as _f:
        _f.write(cutoff_ts)

    total_elapsed = time.time() - t0
    print()
    print(SEP)
    print("TFT-ACB-XML Training complete!")
    print(f"[Training] TFT accuracy:      {tft_acc:.3f}  (ensemble of {tft_model.N_ENSEMBLE} seeds)")
    print(f"[Training] BiLSTM accuracy:   {bilstm_acc:.3f}  (ensemble of {bilstm_model.N_ENSEMBLE} seeds)")
    print(f"[Training] Meta routing acc:  {meta_results['train_acc']:.3f}  (OOF, agreement-based gate)")
    print(f"[Training] Gate CV acc:       {meta_results.get('gate_cv_acc', 0):.3f}  "
          f"(on {meta_results.get('n_disagree', 0):,} disagreement rows, "
          f"{meta_results.get('pct_disagree', 0):.1f}% of OOF)")
    w = meta_results.get("weights", {})
    print(f"[Training] Static fallback — TFT: {w.get('tft',0):.3f}  BiLSTM: {w.get('bilstm',0):.3f}")
    print(f"[Training] Data window: {TRAINING_START} -> {TRAINING_CUTOFF_DATE}")
    print(f"Models saved to:  models/saved/")
    print(f"Total time:       {_fmt(total_elapsed)}")
    print(SEP)


if __name__ == "__main__":
    force       = "--force"  in sys.argv
    test_mode   = "--test"   in sys.argv
    config_only = "--config" in sys.argv

    if config_only:
        import json as _json
        df_tft    = load_tft_data()
        df_bilstm = load_bilstm_data()
        print(f"TFT config:    {len(df_tft):,} rows, {len(TFT_FEAT_COLS)} features, "
              f"HORIZON={tft_model.HORIZON}")
        print(f"BiLSTM config: {len(df_bilstm):,} rows, {len(BILSTM_FEAT_COLS)} features, "
              f"HORIZON={bilstm_model.HORIZON}")
    else:
        run_training(force=force, test_mode=test_mode)
