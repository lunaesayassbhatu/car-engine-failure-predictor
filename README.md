# Car Engine Failure Predictor

A machine learning system that predicts whether a car engine is at risk of failure based on real-time diagnostic parameters — complete with a desktop GUI built in Tkinter.

## How It Works

1. Generates synthetic engine telemetry data (5,000 samples) using realistic distributions
2. Trains a **Random Forest classifier** on labeled failure outcomes
3. Accepts engine parameters through a GUI and outputs a failure probability and risk status

## Features

- Synthetic data generation with physics-informed risk scoring
- Random Forest model with 300 estimators and class balancing
- Desktop GUI — enter engine stats and get an instant prediction
- Model persistence via `joblib`

## Engine Parameters

| Parameter | Unit | Description |
|-----------|------|-------------|
| Mileage | km | Total distance driven |
| Engine Temperature | °C | Current engine temp |
| Oil Pressure | bar | Current oil pressure |
| RPM | rpm | Engine revolutions per minute |
| Error Code Count | count | Number of active error codes |
| Time Since Last Service | days | Days since last maintenance |
| Avg Trip Duration | minutes | Average length of trips |
| City Drive Ratio | 0–1 | Proportion of city vs highway driving |

## Model Performance

- **Accuracy:** 71%
- **Recall (failure class):** 86%
- **Algorithm:** Random Forest (300 trees, balanced class weights)

## Usage

```bash
pip install -r requirements.txt
jupyter notebook engine_failure_predictor.ipynb
```

## Tech Stack

- Python
- scikit-learn
- pandas / NumPy
- Tkinter (GUI)
- joblib
