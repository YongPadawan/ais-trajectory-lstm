"""Step 1: reproduce the thesis experiment and compare it with simple baselines.

The pipeline follows the original notebook as closely as possible:
  - drop every row with any missing value (`df.dropna()`)
  - pick the vessel with the most messages
  - resample its track to 1 / 5 / 10 minutes with linear interpolation
  - MinMax-scale latitude/longitude, windows of 10 steps, 80/20 chronological split
  - same 2-layer LSTM, 50 epochs, batch size 32

New in this version: the same test points are also scored with two baselines
that need no training, and errors are reported in meters as well as degrees.

Run:  python -m ais_traj.reproduce
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from . import data, models

WINDOW = 10
FREQS = ["1min", "5min", "10min"]


def make_windows(arr: np.ndarray, window: int):
    X = np.stack([arr[i : i + window] for i in range(len(arr) - window)])
    y = arr[window:]
    return X, y


def score(name: str, y_true_deg: np.ndarray, y_pred_deg: np.ndarray) -> dict:
    err_m = data.haversine_m(
        y_true_deg[:, 0], y_true_deg[:, 1], y_pred_deg[:, 0], y_pred_deg[:, 1]
    )
    return {
        "method": name,
        "MSE_deg": float(mean_squared_error(y_true_deg, y_pred_deg)),
        "MAE_deg": float(mean_absolute_error(y_true_deg, y_pred_deg)),
        "R2": float(r2_score(y_true_deg, y_pred_deg)),
        "mean_error_m": float(err_m.mean()),
        "median_error_m": float(np.median(err_m)),
        "p90_error_m": float(np.percentile(err_m, 90)),
    }


def thesis_vessel_track(raw: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    df = raw.dropna()
    df = df.sort_values(["MMSI", "# Timestamp"])
    mmsi = int(df["MMSI"].value_counts().idxmax())
    v = df[df["MMSI"] == mmsi].copy()
    v["# Timestamp"] = pd.to_datetime(v["# Timestamp"], dayfirst=True)
    v = v.sort_values("# Timestamp").set_index("# Timestamp")
    return mmsi, v.select_dtypes("number")


def run_one(track: pd.DataFrame, freq: str) -> tuple[list[dict], dict]:
    res = track.resample(freq).mean().interpolate("linear").reset_index()
    latlon = res[["Latitude", "Longitude"]].to_numpy()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(latlon)
    X, y = make_windows(scaled, WINDOW)
    split = int(len(X) * 0.8)
    X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]

    models.set_seed(42)
    model = models.thesis_lstm(WINDOW, 2)
    hist = model.fit(
        X_tr, y_tr, validation_data=(X_te, y_te), epochs=50, batch_size=32, verbose=0
    )

    inv = scaler.inverse_transform
    y_true = inv(y_te)
    X_te_deg = inv(X_te.reshape(-1, 2)).reshape(X_te.shape)
    rows = [
        score("Thesis LSTM", y_true, inv(model.predict(X_te, verbose=0))),
        score("Persistence (stay put)", y_true, models.persistence(X_te_deg)),
        score("Constant velocity", y_true, models.constant_velocity(X_te_deg)),
    ]
    for r in rows:
        r.update(freq=freq, n_train=len(X_tr), n_test=len(X_te))

    # Distance the ship actually moved per step in the test set, for context.
    step_m = data.haversine_m(
        X_te_deg[:, -1, 0], X_te_deg[:, -1, 1], y_true[:, 0], y_true[:, 1]
    )
    extra = {
        "freq": freq,
        "mean_step_m": float(step_m.mean()),
        "loss": hist.history["loss"],
        "val_loss": hist.history["val_loss"],
        "test_true": y_true.tolist(),
        "test_pred": inv(model.predict(X_te, verbose=0)).tolist(),
        "track": latlon.tolist(),
    }
    return rows, extra


def main():
    print(models.gpu_summary())
    print("Loading raw CSV (this takes a minute)...")
    raw = data.load_raw()
    mmsi, track = thesis_vessel_track(raw)
    del raw
    print(f"Thesis vessel: MMSI {mmsi}, {len(track)} messages after dropna")

    all_rows, extras = [], []
    for freq in FREQS:
        rows, extra = run_one(track, freq)
        all_rows += rows
        extras.append(extra)
        print(pd.DataFrame(rows).round(6).to_string(index=False))

    out = data.ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out / "reproduce_metrics.csv", index=False)
    with open(out / "reproduce_details.json", "w") as f:
        json.dump({"mmsi": mmsi, "runs": extras}, f)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
