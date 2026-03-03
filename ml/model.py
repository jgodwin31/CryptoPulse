"""
model.py
--------
XGBoost model for CryptoPulse price direction prediction.
Patched: removed use_label_encoder, added early stopping.

Usage:
    python -m ml.model --symbol BTC --train
    python -m ml.model --symbol BTC --predict
    python -m ml.model --all --predict
"""

import argparse
import logging
import os
import pickle
import sys
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.features import (
    compute_features,
    get_feature_columns,
    load_price_history,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cryptopulse.model")

# ── Config ─────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

PREDICTION_HORIZON = 5
TEST_SIZE          = 0.2
N_CV_SPLITS        = 5

XGBOOST_PARAMS = {
    "n_estimators":          500,
    "max_depth":             4,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "min_child_weight":      5,
    "gamma":                 0.1,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "scale_pos_weight":      1,
    "eval_metric":           "logloss",
    "early_stopping_rounds": 30,
    "random_state":          42,
    "n_jobs":                -1,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

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
    with open(model_path(symbol), "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path(symbol), "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Saved model and scaler for %s → %s", symbol, MODELS_DIR)

def load_artifacts(symbol: str) -> Tuple[XGBClassifier, StandardScaler]:
    mp, sp = model_path(symbol), scaler_path(symbol)
    if not mp.exists() or not sp.exists():
        raise FileNotFoundError(f"No trained model for {symbol}. Run --train first.")
    with open(mp, "rb") as f:
        model = pickle.load(f)
    with open(sp, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# ── Training ───────────────────────────────────────────────────────────────────

def train(symbol: str, engine: Engine):
    logger.info("=" * 50)
    logger.info("Training model for %s", symbol)
    logger.info("=" * 50)

    df_raw  = load_price_history(engine, symbol=symbol, limit=50000)
    df_feat = compute_features(df_raw, horizon=PREDICTION_HORIZON)

    feature_cols = get_feature_columns()
    X = df_feat[feature_cols].values
    y = df_feat["target"].values

    logger.info("Dataset: %d samples | %.1f%% up / %.1f%% down",
                len(y), y.mean()*100, (1-y.mean())*100)

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    XGBOOST_PARAMS["scale_pos_weight"] = round(pos_weight, 2)

    split_idx       = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_scores = []
    logger.info("Running %d-fold time-series CV...", N_CV_SPLITS)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_s), 1):
        xtr, xval = X_train_s[tr_idx], X_train_s[val_idx]
        ytr, yval = y_train[tr_idx],   y_train[val_idx]
        cv_model = XGBClassifier(**XGBOOST_PARAMS)
        cv_model.fit(xtr, ytr, eval_set=[(xval, yval)], verbose=False)
        score = roc_auc_score(yval, cv_model.predict_proba(xval)[:, 1])
        cv_scores.append(score)
        logger.info("  Fold %d AUC: %.4f (best iter: %d)", fold, score, cv_model.best_iteration)

    logger.info("CV AUC: %.4f +/- %.4f", np.mean(cv_scores), np.std(cv_scores))

    # Final model
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=50)
    logger.info("Best iteration: %d", model.best_iteration)

    y_pred   = model.predict(X_test_s)
    y_prob   = model.predict_proba(X_test_s)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob)

    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))
    logger.info("Test AUC: %.4f", test_auc)

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info("\nTop 10 features:\n%s", importances.head(10).to_string())

    save_artifacts(symbol, model, scaler)
    return model, scaler

# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_latest(symbol: str, engine: Engine,
                   model=None, scaler=None) -> dict:
    if model is None or scaler is None:
        model, scaler = load_artifacts(symbol)

    feature_cols = get_feature_columns()
    df_raw  = load_price_history(engine, symbol=symbol, limit=500)
    df_feat = compute_features(df_raw, horizon=PREDICTION_HORIZON)

    if df_feat.empty:
        raise ValueError(f"Not enough data for {symbol}")

    latest       = df_feat[feature_cols].iloc[[-1]]
    latest_price = df_raw["current_price"].iloc[-1]

    X_scaled  = scaler.transform(latest.values)
    prob_up   = float(model.predict_proba(X_scaled)[0, 1])
    prob_down = 1 - prob_up

    if prob_up >= 0.60:
        signal, confidence = "BUY",  prob_up
    elif prob_down >= 0.60:
        signal, confidence = "SELL", prob_down
    else:
        signal, confidence = "HOLD", max(prob_up, prob_down)

    result = {
        "symbol":       symbol,
        "price":        round(latest_price, 4),
        "signal":       signal,
        "prob_up":      round(prob_up, 4),
        "prob_down":    round(prob_down, 4),
        "confidence":   round(confidence, 4),
        "horizon":      PREDICTION_HORIZON,
        "predicted_at": pd.Timestamp.utcnow().isoformat(),
    }
    logger.info("%-6s | $%-10.2f | %-4s | P(up)=%.2f | P(down)=%.2f | Conf=%.2f",
                symbol, latest_price, signal, prob_up, prob_down, confidence)
    return result


def predict_all(symbols: list, engine: Engine) -> pd.DataFrame:
    results = []
    for sym in symbols:
        try:
            results.append(predict_latest(sym, engine))
        except FileNotFoundError:
            logger.warning("No model for %s — run --train first", sym)
        except Exception as e:
            logger.error("Prediction failed for %s: %s", sym, e)
    return pd.DataFrame(results)

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default="BTC")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    engine = get_engine()

    if args.train:
        train(args.symbol.upper(), engine)
    if args.predict:
        print(predict_latest(args.symbol.upper(), engine))
    if args.all:
        SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "DOT", "AVAX"]
        print(predict_all(SYMBOLS, engine).to_string(index=False))