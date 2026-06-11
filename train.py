"""
Main Training Pipeline — Electricity Load Forecasting.
Trains 3 best base models + Hybrid Stacking Ensemble for maximum accuracy.

Models:
  1. XGBoost     — Best at engineered features (lags, calendar, holidays)
  2. LSTM        — Deep learning temporal baseline
  3. CNN-BiLSTM-Attention — SOTA deep learning with attention
  4. Hybrid Ensemble — Stacking meta-learner that blends all 3

Usage: python train.py
"""

import os
import sys
import time
import warnings
import json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import run_preprocessing
from src.feature_engineering import build_features, get_feature_columns
from src.evaluate import (
    compute_metrics, compute_metrics_by_period, diebold_mariano_test,
    print_comparison_table, save_results
)
from src.visualization import (
    plot_load_timeseries, plot_seasonal_patterns,
    plot_actual_vs_predicted, plot_scatter_actual_vs_predicted,
    plot_metrics_comparison, plot_error_by_hour, plot_error_by_season,
    plot_error_distribution, plot_training_history,
    plot_feature_importance, plot_all_models_forecast,
    plot_ensemble_weights
)

# Directories
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Config
SEQ_LENGTH = 168        # 1 week lookback
FORECAST_HORIZON = 24   # 24 hours ahead
BATCH_SIZE = 64
MAX_EPOCHS = 200

# GPU check
if torch.cuda.is_available():
    print(f"[GPU] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("[WARN] CUDA not available — training will use CPU")


def download_data():
    """Download dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOAD DATA")
    print("=" * 60)
    from download_data import download_dataset
    return download_dataset()


def preprocess_data():
    """Run preprocessing and return splits."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)
    train, val, test, scaler = run_preprocessing()
    return train, val, test, scaler


def generate_eda_plots(train, val, test):
    """Generate EDA plots."""
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS PLOTS")
    print("=" * 60)

    full_data = pd.concat([train, val, test])
    plot_load_timeseries(full_data, os.path.join(PLOTS_DIR, "load_timeseries.png"))
    plot_seasonal_patterns(full_data, PLOTS_DIR)


def prepare_ml_data(train, val, test, target_cols):
    """Prepare data for ML models (XGBoost)."""
    print("\n[INFO] Building features for ML models...")

    train_feat = build_features(train.copy(), target_cols=target_cols)
    val_feat = build_features(val.copy(), target_cols=target_cols)
    test_feat = build_features(test.copy(), target_cols=target_cols)

    feature_cols = get_feature_columns(train_feat, target_cols=target_cols)

    X_train = train_feat[feature_cols]
    y_train = train_feat[target_cols]
    X_val = val_feat[feature_cols]
    y_val = val_feat[target_cols]
    X_test = test_feat[feature_cols]
    y_test = test_feat[target_cols]

    # Save feature columns
    with open(os.path.join(DATA_PROCESSED, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    print(f"[INFO] ML features: {len(feature_cols)} columns")
    print(f"[INFO] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, train_feat.index, val_feat.index, test_feat.index


def prepare_dl_data(train, val, test):
    """Prepare data for DL models (LSTM, CNN-BiLSTM-Attention)."""
    print("\n[INFO] Preparing sequences for deep learning models...")

    # Fit scaler on train only
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values)
    val_scaled = scaler.transform(val.values)
    test_scaled = scaler.transform(test.values)

    # Save scaler
    joblib.dump(scaler, os.path.join(DATA_PROCESSED, "dl_scaler.pkl"))

    # Create sequences
    def create_sequences(data, seq_len=SEQ_LENGTH, horizon=FORECAST_HORIZON):
        X, y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            X.append(data[i:i + seq_len])
            # Flatten the multi-target horizon sequence
            y.append(data[i + seq_len:i + seq_len + horizon].flatten())
        return np.array(X), np.array(y)

    X_train_dl, y_train_dl = create_sequences(train_scaled)

    # For validation: prepend last SEQ_LENGTH from train
    val_with_context = np.concatenate([train_scaled[-SEQ_LENGTH:], val_scaled], axis=0)
    X_val_dl, y_val_dl = create_sequences(val_with_context)

    # For test: prepend last SEQ_LENGTH from val
    test_with_context = np.concatenate([val_scaled[-SEQ_LENGTH:], test_scaled], axis=0)
    X_test_dl, y_test_dl = create_sequences(test_with_context)

    print(f"[INFO] DL sequences — Train: {X_train_dl.shape}, Val: {X_val_dl.shape}, Test: {X_test_dl.shape}")

    return X_train_dl, y_train_dl, X_val_dl, y_val_dl, X_test_dl, y_test_dl, scaler


def train_lstm(X_train, y_train, X_val, y_val, X_test, y_test, scaler, n_targets):
    """Train and evaluate LSTM."""
    from src.models.lstm_model import LSTMModel

    model = LSTMModel(seq_length=SEQ_LENGTH, n_features=n_targets, forecast_horizon=FORECAST_HORIZON, n_targets=n_targets)
    model.build_model()
    model.train(X_train, y_train, X_val, y_val, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)

    # Predictions on test
    y_pred_scaled = model.predict(X_test)
    model.save(os.path.join(MODELS_DIR, "lstm_model.pt"))

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, n_targets)).ravel()
    y_true = scaler.inverse_transform(y_test.reshape(-1, n_targets)).ravel()

    # Also get validation predictions (for stacking ensemble)
    y_val_pred_scaled = model.predict(X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled.reshape(-1, n_targets)).ravel()
    y_val_true = scaler.inverse_transform(y_val.reshape(-1, n_targets)).ravel()

    history = model.history

    return y_pred, y_true, y_val_pred, y_val_true, model.train_time, history


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate XGBoost."""
    from src.models.xgboost_model import XGBoostModel

    model = XGBoostModel()
    model.train(X_train, y_train, X_val, y_val, optimize=True)

    y_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)  # For stacking ensemble
    model.save(os.path.join(MODELS_DIR, "xgboost.json"))

    # Feature importance
    importance = model.get_feature_importance(top_n=20)
    if importance:
        plot_feature_importance(importance, os.path.join(PLOTS_DIR, "feature_importance.png"))

    return y_pred, y_val_pred, model.train_time


def train_cnn_bilstm_attention(X_train, y_train, X_val, y_val, X_test, y_test, scaler, n_targets):
    """Train and evaluate CNN-BiLSTM-Attention (SOTA) for Multi-Target forecasting."""
    from src.models.cnn_bilstm_attention import CNNBiLSTMAttentionModel

    model = CNNBiLSTMAttentionModel(
        seq_length=SEQ_LENGTH, n_features=n_targets, forecast_horizon=FORECAST_HORIZON, n_targets=n_targets
    )
    model.build_model()
    model.train(X_train, y_train, X_val, y_val, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE)

    # Predictions on test
    y_pred_scaled = model.predict(X_test)
    model.save(os.path.join(MODELS_DIR, "cnn_bilstm_attention.pt"))

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, n_targets)).ravel()
    y_true = scaler.inverse_transform(y_test.reshape(-1, n_targets)).ravel()

    # Also get validation predictions (for stacking ensemble)
    y_val_pred_scaled = model.predict(X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled.reshape(-1, n_targets)).ravel()
    y_val_true = scaler.inverse_transform(y_val.reshape(-1, n_targets)).ravel()

    history = model.history

    return y_pred, y_true, y_val_pred, y_val_true, model.train_time, history


def train_hybrid_ensemble(val_predictions, y_val_true, test_predictions):
    """Train the hybrid stacking ensemble on validation predictions."""
    from src.models.hybrid_ensemble import HybridEnsembleModel

    model = HybridEnsembleModel()
    model.train(val_predictions, y_val_true)
    model.save(os.path.join(MODELS_DIR, "hybrid_ensemble.pkl"))

    # Generate test predictions
    y_pred = model.predict(test_predictions)

    # Plot ensemble weights
    weights = model.get_weights()
    if weights:
        plot_ensemble_weights(weights, os.path.join(PLOTS_DIR, "ensemble_weights.png"))

    return y_pred, model.train_time


def main():
    """Main training pipeline."""
    total_start = time.time()

    print("=" * 60)
    print("  ELECTRICITY LOAD FORECASTING")
    print("  Multivariate Pan-European Hybrid Ensemble Pipeline")
    print("=" * 60)
    print(f"  Lookback:          {SEQ_LENGTH} hours (1 week)")
    print(f"  Forecast horizon:  {FORECAST_HORIZON} hours")
    print(f"  Target:            Simultaneous 30+ European Countries")
    print(f"  Dataset:           OPSD Time Series")
    print(f"  Models:            XGBoost + LSTM + CNN-BiLSTM-Attention -> Hybrid Ensemble")
    print("=" * 60)

    # ===== Step 1: Download =====
    download_data()

    # ===== Step 2: Preprocess =====
    train, val, test = None, None, None

    # Check if processed data exists
    train_path = os.path.join(DATA_PROCESSED, "train.csv")
    if os.path.exists(train_path):
        print("\n[INFO] Loading preprocessed data...")
        train = pd.read_csv(train_path, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
        val = pd.read_csv(os.path.join(DATA_PROCESSED, "val.csv"), parse_dates=["utc_timestamp"], index_col="utc_timestamp")
        test = pd.read_csv(os.path.join(DATA_PROCESSED, "test.csv"), parse_dates=["utc_timestamp"], index_col="utc_timestamp")
        print(f"[INFO] Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    else:
        train, val, test, _ = preprocess_data()

    target_cols = train.columns.tolist()
    n_targets = len(target_cols)

    # ===== Step 3: EDA Plots =====
    generate_eda_plots(train, val, test)

    # ===== Step 4: Prepare data =====
    X_train_ml, y_train_ml, X_val_ml, y_val_ml, X_test_ml, y_test_ml, train_ts_ml, val_ts_ml, test_ts_ml = prepare_ml_data(train, val, test, target_cols)
    X_train_dl, y_train_dl, X_val_dl, y_val_dl, X_test_dl, y_test_dl, dl_scaler = prepare_dl_data(train, val, test)

    # ===== Step 5: Train base models =====
    all_metrics = []
    all_predictions = {}
    all_errors = {}
    all_period_metrics = {}
    dl_histories = {}
    train_times = {}

    # ----- 5.1: XGBoost -----
    print("\n" + "#" * 60)
    print("# BASE MODEL 1/3: XGBoost")
    print("#" * 60)
    xgb_pred, xgb_val_pred, xgb_time = train_xgboost(
        X_train_ml, y_train_ml, X_val_ml, y_val_ml, X_test_ml, y_test_ml
    )
    xgb_metrics = compute_metrics(y_test_ml, xgb_pred, "XGBoost")
    xgb_metrics["Train Time (s)"] = round(xgb_time, 2)
    all_metrics.append(xgb_metrics)
    all_predictions["XGBoost"] = xgb_pred
    all_errors["XGBoost"] = y_test_ml.values - xgb_pred
    all_period_metrics["XGBoost"] = compute_metrics_by_period(y_test_ml.values, xgb_pred, test_ts_ml)
    train_times["XGBoost"] = xgb_time

    # ----- 5.2: LSTM -----
    print("\n" + "#" * 60)
    print("# BASE MODEL 2/3: LSTM")
    print("#" * 60)
    lstm_pred, lstm_true, lstm_val_pred, lstm_val_true, lstm_time, lstm_hist = train_lstm(
        X_train_dl, y_train_dl, X_val_dl, y_val_dl, X_test_dl, y_test_dl, dl_scaler, n_targets
    )
    lstm_metrics = compute_metrics(lstm_true, lstm_pred, "LSTM")
    lstm_metrics["Train Time (s)"] = round(lstm_time, 2)
    all_metrics.append(lstm_metrics)
    all_predictions["LSTM"] = lstm_pred
    all_errors["LSTM"] = lstm_true - lstm_pred
    dl_histories["LSTM"] = lstm_hist
    train_times["LSTM"] = lstm_time

    dl_test_timestamps = test.index[:len(lstm_pred)]
    all_period_metrics["LSTM"] = compute_metrics_by_period(lstm_true, lstm_pred, dl_test_timestamps)

    # ----- 5.3: CNN-BiLSTM-Attention -----
    print("\n" + "#" * 60)
    print("# BASE MODEL 3/3: CNN-BiLSTM-Attention (SOTA)")
    print("#" * 60)
    cba_pred, cba_true, cba_val_pred, cba_val_true, cba_time, cba_hist = train_cnn_bilstm_attention(
        X_train_dl, y_train_dl, X_val_dl, y_val_dl, X_test_dl, y_test_dl, dl_scaler, n_targets
    )
    cba_metrics = compute_metrics(cba_true, cba_pred, "CNN-BiLSTM-Attention")
    cba_metrics["Train Time (s)"] = round(cba_time, 2)
    all_metrics.append(cba_metrics)
    all_predictions["CNN-BiLSTM-Attention"] = cba_pred
    all_errors["CNN-BiLSTM-Attention"] = cba_true - cba_pred
    dl_histories["CNN-BiLSTM-Attention"] = cba_hist
    train_times["CNN-BiLSTM-Attention"] = cba_time
    all_period_metrics["CNN-BiLSTM-Attention"] = compute_metrics_by_period(cba_true, cba_pred, dl_test_timestamps)

    # ===== Step 6: Hybrid Stacking Ensemble =====
    print("\n" + "#" * 60)
    print("# HYBRID ENSEMBLE (Stacking Meta-Learner)")
    print("#" * 60)

    # We need to align predictions for the ensemble.
    # XGBoost uses ML features (different length), DL models use sequences.
    # For the ensemble, we use only the DL-aligned test points since DL models
    # produce fewer predictions than XGBoost due to sequence creation.
    #
    # Strategy: Use the DL test timestamps to index into XGBoost predictions.
    # XGBoost predicts for every row in the ML test set (which has timestamps).
    # We align by taking the first N XGBoost predictions where N = len(DL predictions).

    n_dl_test = len(lstm_pred)
    n_dl_val = len(lstm_val_pred)

    # Align XGBoost predictions to DL length
    xgb_pred_aligned = xgb_pred[:n_dl_test]
    xgb_val_pred_aligned = xgb_val_pred[:n_dl_val]

    # Validation predictions for training the meta-learner
    val_predictions = {
        "XGBoost": xgb_val_pred_aligned,
        "LSTM": lstm_val_pred,
        "CNN-BiLSTM-Attention": cba_val_pred,
    }

    # Use DL val true values (they should be the same as XGBoost val true after alignment)
    y_val_true_ensemble = lstm_val_true

    # Test predictions for final ensemble prediction
    test_predictions = {
        "XGBoost": xgb_pred_aligned,
        "LSTM": lstm_pred,
        "CNN-BiLSTM-Attention": cba_pred,
    }

    hybrid_pred, hybrid_time = train_hybrid_ensemble(
        val_predictions, y_val_true_ensemble, test_predictions
    )

    # The true values for hybrid = same as DL true values (aligned)
    hybrid_true = lstm_true[:len(hybrid_pred)]

    hybrid_metrics = compute_metrics(hybrid_true, hybrid_pred, "Hybrid Ensemble")
    hybrid_metrics["Train Time (s)"] = round(hybrid_time, 2)
    all_metrics.append(hybrid_metrics)
    all_predictions["Hybrid Ensemble"] = hybrid_pred
    all_errors["Hybrid Ensemble"] = hybrid_true - hybrid_pred
    train_times["Hybrid Ensemble"] = hybrid_time
    hybrid_timestamps = dl_test_timestamps[:len(hybrid_pred)]
    all_period_metrics["Hybrid Ensemble"] = compute_metrics_by_period(
        hybrid_true, hybrid_pred, hybrid_timestamps
    )

    # ===== Step 7: Evaluation & Comparison =====
    print("\n" + "=" * 60)
    print("STEP 7: EVALUATION & COMPARISON")
    print("=" * 60)

    # Print comparison table
    comparison_df = print_comparison_table(all_metrics)

    # Diebold-Mariano test — compare each model against the hybrid ensemble
    dm_results = {}
    best_model = "Hybrid Ensemble"
    best_errors = all_errors.get(best_model)

    if best_errors is not None:
        for model_name, errors in all_errors.items():
            if model_name != best_model:
                min_len = min(len(best_errors), len(errors))
                dm_stat, p_val = diebold_mariano_test(
                    errors[:min_len], best_errors[:min_len], horizon=FORECAST_HORIZON
                )
                dm_results[f"{model_name} vs {best_model}"] = {
                    "dm_stat": float(dm_stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05)
                }
                sig = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
                print(f"  DM test — {model_name} vs {best_model}: stat={dm_stat}, p={p_val} ({sig})")

    # Save all results
    save_results(all_metrics, all_period_metrics, dm_results, RESULTS_DIR)

    # ===== Step 8: Generate All Plots =====
    print("\n" + "=" * 60)
    print("STEP 8: GENERATING PLOTS")
    print("=" * 60)

    # Metrics comparison bar chart
    print("\n[INFO] Generating final visualization reports...")
    plot_metrics_comparison(all_metrics, PLOTS_DIR)

    # Actual vs Predicted for XGBoost
    if "XGBoost" in all_predictions:
        plot_actual_vs_predicted(
            y_test_ml.values, all_predictions["XGBoost"], test_ts_ml,
            "XGBoost", os.path.join(PLOTS_DIR, "actual_vs_pred_xgboost.png")
        )
        plot_scatter_actual_vs_predicted(
            y_test_ml.values, all_predictions["XGBoost"],
            "XGBoost", os.path.join(PLOTS_DIR, "scatter_xgboost.png")
        )

    # DL models + Hybrid
    for model_name, true_vals in [
        ("LSTM", lstm_true),
        ("CNN-BiLSTM-Attention", cba_true),
        ("Hybrid Ensemble", hybrid_true),
    ]:
        if model_name in all_predictions:
            ts = dl_test_timestamps[:len(all_predictions[model_name])]
            plot_actual_vs_predicted(
                true_vals[:len(all_predictions[model_name])],
                all_predictions[model_name], ts,
                model_name, os.path.join(PLOTS_DIR, f"actual_vs_pred_{model_name.lower().replace(' ', '_').replace('-', '_')}.png")
            )
            plot_scatter_actual_vs_predicted(
                true_vals[:len(all_predictions[model_name])],
                all_predictions[model_name],
                model_name, os.path.join(PLOTS_DIR, f"scatter_{model_name.lower().replace(' ', '_').replace('-', '_')}.png")
            )

    # Error distribution
    plot_error_distribution(all_errors, os.path.join(PLOTS_DIR, "error_distribution.png"))

    # Error by hour and season
    plot_error_by_hour(all_period_metrics, os.path.join(PLOTS_DIR, "mape_by_hour.png"))
    plot_error_by_season(all_period_metrics, os.path.join(PLOTS_DIR, "mape_by_season.png"))

    # Training history for DL models
    if dl_histories:
        plot_training_history(dl_histories, os.path.join(PLOTS_DIR, "training_history.png"))

    # All models overlay (2 week sample)
    plot_all_models_forecast(
        hybrid_true,
        {k: v for k, v in all_predictions.items()},
        dl_test_timestamps[:len(hybrid_true)],
        os.path.join(PLOTS_DIR, "all_models_forecast.png"),
        n_points=336
    )

    # ===== Summary =====
    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("  TRAINING COMPLETE!")
    print("=" * 80)

    # Training time breakdown
    print("\n  ⏱  TRAINING TIME BREAKDOWN:")
    print(f"  {'─'*50}")
    for model_name, t in train_times.items():
        mins = t / 60
        if mins >= 1:
            print(f"    {model_name:30s} │ {mins:6.1f} min")
        else:
            print(f"    {model_name:30s} │ {t:6.1f} sec")
    print(f"  {'─'*50}")
    print(f"    {'TOTAL':30s} │ {total_time/60:6.1f} min")

    print(f"\n  Results saved to: {RESULTS_DIR}")
    print(f"  Plots saved to: {PLOTS_DIR}")

    # Final results with accuracy
    print("\n" + "=" * 80)
    print("  FINAL RESULTS:")
    print("=" * 80)
    print(f"  {'Model':30s} │ {'Accuracy':>10s} │ {'MAPE':>8s} │ {'MAE (MW)':>10s} │ {'R²':>8s} │ {'Within 5%':>10s} │ {'Within 10%':>11s}")
    print(f"  {'─'*30}─┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*13}")
    for m in all_metrics:
        print(f"    {m['Model']:30s}│ {m['Accuracy (%)']:8.2f}%  │ {m['MAPE (%)']:6.2f}% │ {m['MAE (MW)']:8.2f}   │ {m['R²']:.4f}  │ {m['Within 5% (%)']:8.2f}%  │ {m['Within 10% (%)']:8.2f}%")

    # Best model
    best = min(all_metrics, key=lambda x: x["MAPE (%)"])
    print(f"\n  🏆 BEST MODEL: {best['Model']}")
    print(f"     Accuracy:     {best['Accuracy (%)']}%")
    print(f"     MAPE:         {best['MAPE (%)']}%")
    print(f"     MAE:          {best['MAE (MW)']} MW")
    print(f"     R²:           {best['R²']}")
    print(f"     Within 5%:    {best['Within 5% (%)']}% of predictions")
    print(f"     Within 10%:   {best['Within 10% (%)']}% of predictions")

    # Show improvement
    base_best = min([m for m in all_metrics if m["Model"] != "Hybrid Ensemble"], key=lambda x: x["MAPE (%)"])
    hybrid_m = [m for m in all_metrics if m["Model"] == "Hybrid Ensemble"][0]
    improvement = base_best["MAPE (%)"] - hybrid_m["MAPE (%)"]
    if improvement > 0:
        print(f"\n  📈 Hybrid Ensemble improved MAPE by {improvement:.2f}% over best base model ({base_best['Model']})")
        acc_improvement = hybrid_m["Accuracy (%)"] - base_best["Accuracy (%)"]
        print(f"  📈 Accuracy improved from {base_best['Accuracy (%)']}% → {hybrid_m['Accuracy (%)']}% (+{acc_improvement:.2f}%)")
    else:
        print(f"\n  ℹ️  Best base model ({base_best['Model']}) is competitive with ensemble")

    print("=" * 80)

    return all_metrics


if __name__ == "__main__":
    results = main()

