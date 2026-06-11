# Electricity Demand Forecasting — Research Project

## Overview

This project implements a comprehensive **electricity demand forecasting** system using the [Open Power System Data (OPSD)](https://data.open-power-system-data.org/time_series/2020-10-06) time series dataset. It compares 6 different models ranging from simple statistical baselines to state-of-the-art deep learning architectures for **24-hour ahead** load prediction of Germany's electricity demand.

## Dataset

- **Source**: Open Power System Data — Time Series (2020-10-06 release)
- **Resolution**: Hourly (60-minute intervals)
- **Target**: Germany (DE) actual electricity load (MW) from ENTSO-E Transparency
- **Period**: 2015–2020
- **Splits**:
  - Train: 2015-01-01 → 2018-12-31 (~35,000 samples)
  - Validation: 2019-01-01 → 2019-06-30 (~4,300 samples)
  - Test: 2019-07-01 → 2020-06-30 (~8,700 samples)

## Models

| # | Model | Type | Description |
|---|-------|------|-------------|
| 1 | **Linear Regression** | Statistical | Simplest baseline — performance floor |
| 2 | **SARIMA** | Statistical | Seasonal ARIMA — captures trend & seasonality |
| 3 | **LSTM** | Deep Learning | 2-layer LSTM — sequence modeling baseline |
| 4 | **GRU** | Deep Learning | 2-layer GRU — lightweight RNN comparison |
| 5 | **XGBoost** | Machine Learning | Gradient boosting with Optuna tuning |
| 6 | **CNN-BiLSTM-Attention** | Deep Learning (SOTA) | Hybrid: Conv1D + BiLSTM + Multi-Head Attention |

## Evaluation Metrics

- **MAE** — Mean Absolute Error (MW)
- **RMSE** — Root Mean Squared Error (MW)
- **MAPE** — Mean Absolute Percentage Error (%)
- **R²** — Coefficient of Determination
- **Max Error** — Worst-case prediction error (MW)
- **Median AE** — Median Absolute Error (MW)
- **NRMSE** — Normalized RMSE (%)
- **Diebold-Mariano Test** — Statistical significance testing between models

## Setup & Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (RTX 3050)
The project is optimized for NVIDIA RTX 3050 (6GB VRAM). TensorFlow will automatically detect and use the GPU. Ensure CUDA and cuDNN are properly installed.

## Usage

### Run Full Pipeline
```bash
python train.py
```

This will:
1. Download the dataset (~500 MB)
2. Preprocess and split the data
3. Generate EDA plots
4. Train all 6 models
5. Evaluate with comprehensive metrics
6. Generate publication-quality plots
7. Output comparison tables and final results

### Download Data Only
```bash
python download_data.py
```

## Project Structure

```
electricity-load-forecasting/
├── data/
│   ├── raw/                          # Downloaded OPSD dataset
│   └── processed/                    # Cleaned, split data
├── src/
│   ├── data_preprocessing.py         # Data cleaning & splitting
│   ├── feature_engineering.py        # Feature creation (40+ features)
│   ├── evaluate.py                   # Comprehensive metrics
│   ├── visualization.py              # Publication-quality plots
│   └── models/
│       ├── linear_regression.py      # Baseline
│       ├── sarima_model.py           # Statistical baseline
│       ├── lstm_model.py             # LSTM
│       ├── gru_model.py              # GRU
│       ├── xgboost_model.py          # XGBoost + Optuna
│       └── cnn_bilstm_attention.py   # SOTA model
├── models/                           # Saved trained models
├── outputs/
│   ├── plots/                        # All generated plots (300 DPI)
│   └── results/                      # Metrics, LaTeX tables, reports
├── train.py                          # Main training pipeline
├── download_data.py                  # Data download script
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## Outputs

After training, find results in:
- `outputs/results/model_comparison.csv` — Full metrics table
- `outputs/results/model_comparison.tex` — LaTeX table for paper
- `outputs/results/evaluation_report.txt` — Detailed text report
- `outputs/results/dm_test_results.json` — Statistical significance tests
- `outputs/plots/` — All publication-quality plots (300 DPI PNG)

## References

- OPSD Dataset: https://doi.org/10.25832/time_series/2020-10-06
- ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
