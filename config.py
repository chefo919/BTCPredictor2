"""
Central configuration for BTCPredictor2.
All hyperparameters, horizons, and trading constants live here.
Change a value once — it propagates everywhere automatically.
"""

# ── Sequence lengths (lookback windows) ──────────────────────────────────────
SEQ_LEN_TFT    = 180    # 180 × 4h steps = 30 days macro context (4h-sampled tft_merged)
SEQ_LEN_BILSTM = 192    # 192 × 15min steps = 48h short-term momentum (15min-sampled bilstm_merged)

# ── Prediction horizons ───────────────────────────────────────────────────────
HORIZON_TFT    = 1440   # 1 day — macro model predicts 1-day direction
HORIZON_BILSTM = 1440   # 1 day — ACB model predicts 1-day direction
HORIZON_META   = 1440   # 1 day — meta actionable signal

# ── Sampling intervals for pre-sampled merged CSVs ───────────────────────────
BILSTM_SAMPLE_INTERVAL = 15    # minutes between rows in bilstm_merged
TFT_SAMPLE_INTERVAL    = 240   # minutes between rows in tft_merged (4 hours)

# ── TFT architecture ──────────────────────────────────────────────────────────
D_MODEL        = 64    # was 32 — 4x more parameters, same memory (attention is seq×seq not d_model)
N_ENSEMBLE     = 3     # seeds per model — predictions averaged for XGBoost and inference
N_HEADS        = 4
DROPOUT_TFT    = 0.3
N_EPOCHS_TFT   = 100
BATCH_TFT      = 32     # CPU-safe. Colab overrides to 512.
STRIDE_TFT     = 1      # 4h-sampled data — stride 1 = one sequence per 4h step

# ── BiLSTM architecture ───────────────────────────────────────────────────────
LSTM_UNITS_BILSTM = 64    # units per direction (64×2=128 total); was hardcoded 32 — underfitting
DROPOUT_BILSTM    = 0.25  # was 0.5 — reduced to fix underfitting
N_EPOCHS_BILSTM   = 100
BATCH_BILSTM      = 16    # CPU-safe. Colab overrides to 512.
STRIDE_BILSTM     = 4     # 1 sample per hour = every 4 steps in 15min-sampled data

# ── Training ──────────────────────────────────────────────────────────────────
TRAINING_CUTOFF_DATE = "2026-04-15"
EXP_WEIGHT_HALFLIFE  = 0.33    # fraction of train set back from end where weight = 0.5

# ── Trading signals ───────────────────────────────────────────────────────────
BUY_THRESHOLD  = 0.54   # static fallback (overridden by _dynamic_thresholds when h4_atr_norm available)
SELL_THRESHOLD = 0.46

# ── Dynamic threshold parameters (Fix 5) ──────────────────────────────────────
VOL_LOW_ATR    = 0.003  # h4_atr_norm below this → quiet market, require high confidence
VOL_HIGH_ATR   = 0.010  # h4_atr_norm above this → volatile market, fee easily outpaced
THRESH_LOW_VOL  = 0.60   # buy threshold in quiet market
THRESH_HIGH_VOL = 0.52   # buy threshold in volatile market

# ── Risk management ───────────────────────────────────────────────────────────
STOP_LOSS_PCT   = 0.03    # 3%
TAKE_PROFIT_PCT = 0.06    # 6%
MIN_HOLD_MIN    = 1440    # minimum hold = one full prediction horizon (1 day)
MAX_HOLD_MIN    = 4320    # 3 days — let signal drive the exit, not the clock
CHOP_FILTER_PCT = 0.005   # skip entries when 1H range < 0.5%

# ── Portfolio ─────────────────────────────────────────────────────────────────
FEE_RATE      = 0.001     # 0.1% per trade (Coinbase Advanced)
STARTING_USDT = 1000.0
