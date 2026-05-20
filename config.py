"""
Central configuration for BTCPredictor2.
All hyperparameters, horizons, and trading constants live here.
Change a value once — it propagates everywhere automatically.
"""

# ── Sequence lengths (lookback windows) ──────────────────────────────────────
SEQ_LEN_TFT    = 1440   # 24 hours — macro context window (h1/h4/d1 features)
SEQ_LEN_BILSTM = 240    # 4 hours  — short-term momentum window (1m/15m/30m features)

# ── Prediction horizons ───────────────────────────────────────────────────────
HORIZON_TFT    = 60     # 1 hour — macro model predicts 1h direction
HORIZON_BILSTM = 60     # 1 hour — ACB model predicts 1h direction
HORIZON_META   = 60     # 1 hour — meta actionable signal

# ── TFT architecture ──────────────────────────────────────────────────────────
D_MODEL        = 32
N_HEADS        = 4
DROPOUT_TFT    = 0.3
N_EPOCHS_TFT   = 100
BATCH_TFT      = 4      # CPU-safe: attention [4,4,1440,1440] = ~134 MB. Colab overrides to 256.
STRIDE_TFT     = 60     # one prediction horizon per training step

# ── BiLSTM architecture ───────────────────────────────────────────────────────
DROPOUT_BILSTM  = 0.5
N_EPOCHS_BILSTM = 100
BATCH_BILSTM    = 16    # CPU-safe for SEQ_LEN=720. Colab overrides to 512.
STRIDE_BILSTM   = 60    # one prediction horizon per training step

# ── Training ──────────────────────────────────────────────────────────────────
TRAINING_CUTOFF_DATE = "2026-04-15"

# ── Trading signals ───────────────────────────────────────────────────────────
BUY_THRESHOLD  = 0.65
SELL_THRESHOLD = 0.40

# ── Risk management ───────────────────────────────────────────────────────────
STOP_LOSS_PCT   = 0.03    # 3%
TAKE_PROFIT_PCT = 0.06    # 6%
MIN_HOLD_MIN    = 60      # minimum hold = one full prediction horizon (1 hour)
MAX_HOLD_MIN    = 720     # 12 hours — let signal drive the exit, not the clock
CHOP_FILTER_PCT = 0.005   # skip entries when 1H range < 0.5%

# ── Portfolio ─────────────────────────────────────────────────────────────────
FEE_RATE      = 0.001     # 0.1% per trade (Coinbase Advanced)
STARTING_USDT = 1000.0
