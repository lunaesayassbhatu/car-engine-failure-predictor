"""
Car Engine Failure Predictor — Random Forest + Desktop GUI
Author: Luna Sbahtu | Arizona State University CSE 572 Data Mining

Predicts whether a car engine is at risk of failure based on
diagnostic parameters. Includes a Tkinter desktop GUI.
"""

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# ─────────────────────────────────────────────
# 1. Data Generation
# ─────────────────────────────────────────────
def generate_engine_data(n=5000, seed=42):
    np.random.seed(seed)
    mileage                 = np.random.uniform(10000, 300000, n)
    engine_temp             = np.clip(np.random.normal(90, 10, n), 60, 130)
    oil_pressure            = np.clip(np.random.normal(3.0, 0.7, n), 0.5, 6.0)
    rpm                     = np.random.uniform(600, 4500, n)
    error_code_count        = np.clip(np.random.poisson(1.0, n), 0, 10)
    time_since_last_service = np.random.uniform(0, 700, n)
    avg_trip_duration       = np.random.uniform(5, 90, n)
    city_drive_ratio        = np.random.uniform(0, 1, n)

    risk = (
        (mileage - 100000) / 100000 +
        (engine_temp - 90) / 20 +
        (3.0 - oil_pressure) / 1.5 +
        error_code_count * 0.5 +
        (time_since_last_service - 365) / 365 +
        city_drive_ratio * 0.5
    )
    prob = 1 / (1 + np.exp(-(risk - 0.5)))
    will_fail = np.random.binomial(1, prob)

    return pd.DataFrame({
        "mileage": mileage, "engine_temp": engine_temp,
        "oil_pressure": oil_pressure, "rpm": rpm,
        "error_code_count": error_code_count,
        "time_since_last_service": time_since_last_service,
        "avg_trip_duration": avg_trip_duration,
        "city_drive_ratio": city_drive_ratio,
        "will_fail_soon": will_fail
    })


# ─────────────────────────────────────────────
# 2. Train Model
# ─────────────────────────────────────────────
def train_model(df):
    X = df.drop("will_fail_soon", axis=1)
    y = df["will_fail_soon"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=300, random_state=42,
        n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Model Performance ===")
    print(f"Accuracy: {(y_pred == y_test).mean():.2%}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "engine_failure_predictor.joblib")
    return model


# ─────────────────────────────────────────────
# 3. Predict
# ─────────────────────────────────────────────
def predict(model, mileage, engine_temp, oil_pressure, rpm,
            error_code_count, time_since_last_service,
            avg_trip_duration, city_drive_ratio, threshold=0.7):
    inp = pd.DataFrame([{
        "mileage": mileage, "engine_temp": engine_temp,
        "oil_pressure": oil_pressure, "rpm": rpm,
        "error_code_count": error_code_count,
        "time_since_last_service": time_since_last_service,
        "avg_trip_duration": avg_trip_duration,
        "city_drive_ratio": city_drive_ratio
    }])
    prob = model.predict_proba(inp)[0][1]
    status = "RISK" if prob >= threshold else "OK"
    msg = ("⚠️ Engine at higher risk of failure. Recommend inspection."
           if status == "RISK" else
           "✅ Engine is unlikely to fail in the near future.")
    return prob, status, msg


# ─────────────────────────────────────────────
# 4. Tkinter GUI
# ─────────────────────────────────────────────
def launch_gui(model):
    root = tk.Tk()
    root.title("Car Engine Failure Predictor")
    root.geometry("480x530")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=15)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Car Engine Failure Predictor",
              font=("Segoe UI", 15, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10))

    fields = [
        ("Mileage (km):", "150000"),
        ("Engine Temp (°C):", "95"),
        ("Oil Pressure (bar):", "2.8"),
        ("RPM:", "2500"),
        ("Error Code Count:", "1"),
        ("Time Since Last Service (days):", "300"),
        ("Avg Trip Duration (min):", "30"),
        ("City Drive Ratio (0–1):", "0.6"),
    ]

    entries = []
    for i, (label, default) in enumerate(fields, start=1):
        ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", pady=3)
        e = ttk.Entry(frame, width=20)
        e.insert(0, default)
        e.grid(row=i, column=1, sticky="w", pady=3)
        entries.append(e)

    result_frame = ttk.LabelFrame(frame, text="Prediction Result", padding=10)
    result_frame.grid(row=10, column=0, columnspan=2, pady=12, sticky="we")

    prob_var   = tk.StringVar(value="Probability: —")
    status_var = tk.StringVar(value="Status: —")
    msg_var    = tk.StringVar(value="")

    ttk.Label(result_frame, textvariable=prob_var,   font=("Segoe UI", 10, "bold")).pack(anchor="w")
    ttk.Label(result_frame, textvariable=status_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")
    ttk.Label(result_frame, textvariable=msg_var,    font=("Segoe UI", 10), wraplength=430).pack(anchor="w")

    def on_predict():
        try:
            vals = [float(e.get()) for e in entries]
            if not 0 <= vals[7] <= 1:
                raise ValueError("City drive ratio must be between 0 and 1")
            prob, status, msg = predict(model, *vals)
            prob_var.set(f"Probability of failure: {prob:.2%}")
            status_var.set(f"Status: {status}")
            msg_var.set(msg)
        except ValueError as ex:
            messagebox.showerror("Input Error", str(ex))

    ttk.Button(frame, text="Predict", command=on_predict).grid(
        row=11, column=0, columnspan=2, pady=8)

    root.mainloop()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating data and training model...")
    df    = generate_engine_data()
    model = train_model(df)
    print("Launching GUI...")
    launch_gui(model)
