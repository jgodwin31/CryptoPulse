"""
model.py
--------
XGBoost model for CryptoPulse price direction prediction.

Trains a binary classifier to predict whether price will be
higher (1) or lower (0) N periods from now, based on technical indicators.

Usage:
    # Train and save model for BTC
    python model.py --symbol BTC --train

    # Load and predict on latest data
    python model.py --symbol BTC --predict
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from xgboost import XGBClassifier

from features import (
    compute_features,
    get_feature_columns,
    load_price_history,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cryptopulse.model")

# ── Config ────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

PREDICTION_HORIZON = 5      # periods ahead to predict
TEST_SIZE          = 0.2    # fraction of data held out for evaluation
N_CV_SPLITS        = 5      # TimeSeriesSplit folds for cross-validation

XGBOOST_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 1,    # adjusted during training if classes imbalanced
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_engine() -> Engine:
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
        f"@{os.getenv('PGHOST','localhost')}:{os.getenv('PGPORT',5432)}/{os.getenv('PGDATABASE')}",
        future=True
    )


def model_path(symbol: str) -> Path:
    return MODELS_DIR / f"{symbol.upper()}_model.pkl"


def scaler_path(symbol: str) -> Path:
    return MODELS_DIR / f"{symbol.upper()}_scaler.pkl"


def save_artifacts(symbol: str, model: XGBClassifier, scaler: StandardScaler):
    with open(model_path(symbol),  "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path(symbol), "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved model and scaler for %s → %s", symbol, MODELS_DIR)


def load_artifacts(symbol: str) -> Tuple[XGBClassifier, StandardScaler]:
    mp, sp = model_path(symbol), scaler_path(symbol)
    if not mp.exists() or not sp.exists():
        raise FileNotFoundError(
            f"No trained model found for {symbol}. Run with --train first."
        )
    with open(mp, "rb") as f:
        model = pickle.load(f)
    with open(sp, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Loaded model and scaler for %s", symbol)
    return model, scaler

# ── Training ──────────────────────────────────────────────────────────────────

def train(symbol: str, engine: Engine) -> Tuple[XGBClassifier, StandardScaler, dict]:
    """
    Full training pipeline:
    1. Load price history
    2. Compute features
    3. Time-series cross-validation
    4. Train final model on all data
    5. Save artifacts

    Returns (model, scaler, metrics_dict)
    """
    logger.info("═" * 50)
    logger.info("Training model for %s", symbol)
    logger.info("═" * 50)

    # 1. Data
    df_raw  = load_price_history(engine, symbol=symbol, limit=5000)
    df_feat = compute_features(df_raw, horizon=PREDICTION_HORIZON)

    feature_cols = get_feature_columns()
    X = df_feat[feature_cols].values
    y = df_feat["target"].values

    logger.info("Dataset: %d samples | Class balance: %.1f%% up, %.1f%% down",
                len(y), y.mean()*100, (1-y.mean())*100)

    # Adjust for class imbalance
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    XGBOOST_PARAMS["scale_pos_weight"] = round(pos_weight, 2)

    # 2. Time-series split (no shuffling — respect temporal order)
    split_idx  = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 3. Scale features
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4. Cross-validation (time-series aware)
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = []
    logger.info("Running %d-fold time-series cross-validation...", N_CV_SPLITS)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_s), 1):
        xtr, xval = X_train_s[tr_idx], X_train_s[val_idx]
        ytr, yval = y_train[tr_idx],   y_train[val_idx]
        cv_model = XGBClassifier(**XGBOOST_PARAMS)
        cv_model.fit(xtr, ytr, eval_set=[(xval, yval)], verbose=False)
        score = roc_auc_score(yval, cv_model.predict_proba(xval)[:, 1])
        cv_scores.append(score)
        logger.info("  Fold %d AUC: %.4f", fold, score)

    logger.info("CV AUC: %.4f ± %.4f", np.mean(cv_scores), np.std(cv_scores))

    # 5. Final model on full training data
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=50,
    )

    # 6. Evaluate on held-out test set
    y_pred      = model.predict(X_test_s)
    y_prob      = model.predict_proba(X_test_s)[:, 1]
    test_auc    = roc_auc_score(y_test, y_prob)

    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
    logger.info("Test AUC: %.4f", test_auc)

    # 7. Feature importance
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    logger.info("\nTop 10 features:\n%s", importances.head(10).to_string())

    metrics = {
        "symbol":        symbol,
        "cv_auc_mean":   round(float(np.mean(cv_scores)), 4),
        "cv_auc_std":    round(float(np.std(cv_scores)), 4),
        "test_auc":      round(float(test_auc), 4),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "horizon":       PREDICTION_HORIZON,
    }

    save_artifacts(symbol, model, scaler)
    return model, scaler, metrics

# ── Prediction ────────────────────────────────────────────────────────────────

def predict_latest(
    symbol: str,
    engine: Engine,
    model: Optional[XGBClassifier] = None,
    scaler: Optional[StandardScaler] = None,
) -> dict:
    """
    Load the latest price data, compute features, and return a prediction
    for the next PREDICTION_HORIZON periods.

    Returns dict with:
        symbol, signal (BUY/SELL/HOLD), probability, confidence, price
    """
    if model is None or scaler is None:
        model, scaler = load_artifacts(symbol)

    feature_cols = get_feature_columns()

    df_raw  = load_price_history(engine, symbol=symbol, limit=200)
    df_feat = compute_features(df_raw, horizon=PREDICTION_HORIZON)

    if df_feat.empty:
        raise ValueError(f"Not enough data to predict for {symbol}")

    # Use the most recent row
    latest = df_feat[feature_cols].iloc[[-1]]
    latest_price = df_raw["current_price"].iloc[-1]

    X_scaled = scaler.transform(latest.values)
    prob_up   = float(model.predict_proba(X_scaled)[0, 1])
    prob_down = 1 - prob_up

    # Signal thresholds
    if prob_up >= 0.60:
        signal     = "BUY"
        confidence = prob_up
    elif prob_down >= 0.60:
        signal     = "SELL"
        confidence = prob_down
    else:
        signal     = "HOLD"
        confidence = max(prob_up, prob_down)

    result = {
        "symbol":      symbol,
        "price":       round(latest_price, 4),
        "signal":      signal,
        "prob_up":     round(prob_up, 4),
        "prob_down":   round(prob_down, 4),
        "confidence":  round(confidence, 4),
        "horizon":     PREDICTION_HORIZON,
        "predicted_at": pd.Timestamp.utcnow().isoformat(),
    }

    logger.info(
        "%-6s | $%-10.2f | %-4s | P(up)=%.2f | P(down)=%.2f | Confidence=%.2f",
        symbol, latest_price, signal, prob_up, prob_down, confidence
    )
    return result


def predict_all(symbols: list, engine: Engine) -> pd.DataFrame:
    """Run predictions for all symbols and return a summary DataFrame."""
    results = []
    for sym in symbols:
        try:
            results.append(predict_latest(sym, engine))
        except FileNotFoundError:
            logger.warning("No model for %s — run --train first", sym)
        except Exception as e:
            logger.error("Prediction failed for %s: %s", sym, e)
    return pd.DataFrame(results)

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CryptoPulse ML Model")
    parser.add_argument("--symbol", default="BTC", help="Crypto symbol (e.g. BTC, ETH)")
    parser.add_argument("--train",  action="store_true", help="Train and save model")
    parser.add_argument("--predict", action="store_true", help="Predict from saved model")
    parser.add_argument("--all",    action="store_true", help="Predict for all tracked symbols")
    args = parser.parse_args()

    engine = get_engine()

    if args.train:
        train(args.symbol.upper(), engine)

    if args.predict:
        result = predict_latest(args.symbol.upper(), engine)
        print(result)

    if args.all:
        from ingest_binance import SYMBOLS
        syms = [s.replace("usdt", "").upper() for s in SYMBOLS]
        df = predict_all(syms, engine)
        print(df.to_string(index=False))