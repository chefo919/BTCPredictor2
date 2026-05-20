from sklearnex import patch_sklearn
patch_sklearn()

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["DNNL_VERBOSE"]          = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
try:
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
except RuntimeError:
    pass  # already initialized (e.g. imported after another TF module)

from features.engineer import FEATURE_COLS, get_feature_groups
from models import tft_model, bilstm_model, meta_model

ROOT        = os.path.dirname(os.path.dirname(__file__))
MERGED_PATH = os.path.join(ROOT, "data", "btc_merged_features.csv")
STAMP_PATH  = os.path.join(ROOT, "models", "saved", "last_train_rows.txt")

from config import TRAINING_CUTOFF_DATE


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


def load_and_engineer() -> pd.DataFrame:
    if not os.path.exists(MERGED_PATH):
        print("btc_merged_features.csv not found. Run features/engineer.py first.", flush=True)
        sys.exit(1)
    print(f"Loading {MERGED_PATH}...", flush=True)
    df = pd.read_csv(MERGED_PATH, parse_dates=["time"])
    print(f"Loaded {len(df):,} rows with {len(FEATURE_COLS)} features.", flush=True)
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
    df = load_and_engineer()

    cutoff_ts = pd.Timestamp(TRAINING_CUTOFF_DATE, tz="UTC")
    before    = len(df)
    df        = df[df["time"] <= cutoff_ts].copy()
    print(f"[Training] Cutoff {TRAINING_CUTOFF_DATE}: {before:,} → {len(df):,} rows", flush=True)
    current_rows = len(df)

    if not force:
        print("Use --force to run training. Skipping.")
        return

    if test_mode:
        df = df.tail(10000).copy()
        print(f"[TEST MODE] Limiting to last {len(df):,} rows, 3 epochs.", flush=True)

    # ── Feature groups — each model gets a specialized subset ────────────────
    groups           = get_feature_groups()
    BILSTM_FEATURES  = groups["bilstm"]        # 1m/15m/30m — 30 features
    TFT_DYN_FEATURES = groups["tft_dynamic"]   # h1/h4/d1   — 30 features
    TFT_STA_FEATURES = groups["tft_static"]    # w1/mo1     — 27 features
    ALL_NEEDED       = BILSTM_FEATURES + TFT_DYN_FEATURES + TFT_STA_FEATURES

    n_clean = len(df.dropna(subset=ALL_NEEDED))
    t0      = time.time()

    SEP = "=" * 46
    print()
    print(SEP)
    print("TFT-ACB-XML STACKING TRAINING (arXiv 2602.12380)")
    print(SEP)
    print(f"Phase 1/3: TFT  (macro, SEQ_LEN={tft_model.SEQ_LEN}, dynamic={len(TFT_DYN_FEATURES)}, static={len(TFT_STA_FEATURES)})")
    print(f"Phase 2/3: BiLSTM/ACB (micro, SEQ_LEN={bilstm_model.SEQ_LEN}, features={len(BILSTM_FEATURES)})")
    print(f"Phase 3/3: XGBoost meta-learner (error-reciprocal weighting)")
    print(f"Training rows:   {n_clean:,}")
    print(f"Target:          price direction {tft_model.HORIZON} minutes ahead")
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

    # ── Phase 1: Train TFT on 70% ────────────────────────────────────────────
    if tft_model.is_trained():
        tft_acc     = _saved_acc.get("tft", 0.51)
        tft_val_err = 1.0 - _saved_acc.get("tft_val", tft_acc)
        print(f"[1/3] TFT already trained — skipping  (saved acc: {tft_acc:.3f})")
    else:
        print(f"[1/3] Training TFT (macro, SEQ_LEN={tft_model.SEQ_LEN}, HORIZON={tft_model.HORIZON})...")
        t_tft       = time.time()
        tft_results = tft_model.train(df, TFT_DYN_FEATURES, TFT_STA_FEATURES)
        tft_elapsed = time.time() - t_tft
        tft_acc     = tft_results["test_acc"]
        tft_val_err = 1.0 - tft_results.get("val_acc", tft_acc)
        print(f"\nTFT complete | Accuracy: {tft_acc:.3f} | Val error: {tft_val_err:.4f} | Time: {_fmt(tft_elapsed)}")
    print()

    # ── Phase 2: Train BiLSTM/ACB ─────────────────────────────────────────────
    if bilstm_model.is_trained():
        bilstm_acc     = _saved_acc.get("bilstm", 0.51)
        bilstm_val_err = 1.0 - _saved_acc.get("bilstm_val", bilstm_acc)
        print(f"[2/3] BiLSTM already trained — skipping  (saved acc: {bilstm_acc:.3f})")
    else:
        print(f"[2/3] Training BiLSTM/ACB (micro, SEQ_LEN={bilstm_model.SEQ_LEN}, HORIZON={bilstm_model.HORIZON})...")
        t_bilstm       = time.time()
        bilstm_results = bilstm_model.train(df, BILSTM_FEATURES)
        bilstm_elapsed = time.time() - t_bilstm
        bilstm_acc     = bilstm_results["test_acc"]
        bilstm_val_err = 1.0 - bilstm_results.get("val_acc", bilstm_acc)
        print(f"\nBiLSTM complete | Accuracy: {bilstm_acc:.3f} | Val error: {bilstm_val_err:.4f} | Time: {_fmt(bilstm_elapsed)}")
    print()

    # ── Phase 3: OOF predictions → train meta ────────────────────────────────
    print("[3/3] Generating OOF predictions and training XGBoost meta-learner...")
    df_clean  = df.dropna(subset=ALL_NEEDED).reset_index(drop=True)
    n_total   = len(df_clean)
    val_start = int(n_total * 0.70)
    val_end   = int(n_total * 0.85)
    df_val    = df_clean.iloc[val_start:val_end].copy()

    val_X_dyn  = df_val[TFT_DYN_FEATURES].values.astype("float32")
    val_X_sta  = df_val[TFT_STA_FEATURES].values.astype("float32")
    val_X_bi   = df_val[BILSTM_FEATURES].values.astype("float32")

    print(f"  Generating TFT OOF predictions on {len(df_val):,} rows...", flush=True)
    val_tft_probs    = tft_model.predict_proba_batch(val_X_dyn, val_X_sta)
    print(f"  Generating BiLSTM OOF predictions on {len(df_val):,} rows...", flush=True)
    val_bilstm_probs = bilstm_model.predict_proba_batch(val_X_bi)

    t_meta       = time.time()
    meta_results = meta_model.train(df_val, val_tft_probs, val_bilstm_probs,
                                     tft_val_err, bilstm_val_err,
                                     TFT_DYN_FEATURES + TFT_STA_FEATURES)
    meta_elapsed = time.time() - t_meta
    print(f"\nMeta-learner complete | Train acc: {meta_results['train_acc']:.3f} | "
          f"Time: {_fmt(meta_elapsed)}")

    save_train_rows(current_rows)

    # ── Save accuracy and cutoff ──────────────────────────────────────────────
    import json as _json
    acc_path = os.path.join(ROOT, "models", "saved", "model_accuracies.json")
    with open(acc_path, "w") as _f:
        _json.dump({"tft": tft_acc, "tft_val": 1.0 - tft_val_err,
                    "bilstm": bilstm_acc, "bilstm_val": 1.0 - bilstm_val_err,
                    "meta": meta_results["train_acc"]}, _f, indent=2)

    cutoff_ts   = str(df["time"].iloc[-1])
    cutoff_path = os.path.join(ROOT, "models", "saved", "training_cutoff.txt")
    with open(cutoff_path, "w") as _f:
        _f.write(cutoff_ts)

    total_elapsed = time.time() - t0
    print()
    print(SEP)
    print("TFT-ACB-XML Training complete!")
    print(f"[Training] TFT accuracy:    {tft_acc:.3f}")
    print(f"[Training] BiLSTM accuracy: {bilstm_acc:.3f}")
    print(f"[Training] Meta accuracy:   {meta_results['train_acc']:.3f}")
    w = meta_results.get("weights", {})
    print(f"[Training] Error-reciprocal weights — TFT: {w.get('tft',0):.3f}  BiLSTM: {w.get('bilstm',0):.3f}")
    print(f"[Training] Data cutoff: {cutoff_ts}")
    print(f"Models saved to:  models/saved/")
    print(f"Total time:       {_fmt(total_elapsed)}")
    print(SEP)


if __name__ == "__main__":
    force       = "--force"  in sys.argv
    test_mode   = "--test"   in sys.argv
    config_only = "--config" in sys.argv

    if config_only:
        import json as _json
        df        = load_and_engineer()
        cutoff_ts = pd.Timestamp(TRAINING_CUTOFF_DATE, tz="UTC")
        df        = df[df["time"] <= cutoff_ts]
        _sel_path = os.path.join(ROOT, "models", "saved", "selected_features.json")
        active    = (_json.load(open(_sel_path))["selected"]
                     if os.path.exists(_sel_path) else FEATURE_COLS)
        n_clean   = len(df.dropna(subset=active))
        print(f"Adaptive TFT config: {n_clean:,} rows, {len(active)} features, HORIZON={tft_model.HORIZON}")
    else:
        run_training(force=force, test_mode=test_mode)
