"""Step 2: an improved, multi-vessel version of the experiment.

What changes compared with the thesis:
  - Uses every moving Class A vessel of the day, not a single ship.
  - Keeps rows with missing SOG/COG instead of dropping 87% of the data.
  - Splits tracks at reporting gaps instead of interpolating across them.
  - Splits train / validation / test **by vessel**, so the model is tested on
    ships it has never seen.
  - Predicts the next displacement in meters (works for any ship, anywhere)
    from past displacements plus speed and course over ground.
  - Compared with three baselines: persistence, constant velocity, and dead
    reckoning from the reported speed and course.

Run:  python -m ais_traj.experiment
"""

import json

import numpy as np
import pandas as pd
from tensorflow import keras

from . import data, models

WINDOW = 10
FREQS = ["1min", "5min", "10min"]
MIN_MEAN_SOG_KN = 2.0  # only score segments where the ship is actually under way
KN_TO_MS = 0.514444
M_PER_DEG_LAT = 111_320.0


def to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
    """Project to a local flat x/y grid in meters (fine over a few hundred km)."""
    x = (lon - lon0) * M_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (lat - lat0) * M_PER_DEG_LAT
    return x, y


def build_samples(segments: list[pd.DataFrame], step_s: float):
    """Turn resampled segments into (input window, target displacement) pairs."""
    X, y, last_xy, dr, mmsis = [], [], [], [], []
    for seg in segments:
        if seg["sog"].mean() < MIN_MEAN_SOG_KN:
            continue
        lat, lon = seg["lat"].to_numpy(), seg["lon"].to_numpy()
        x, yy = to_local_xy(lat, lon, lat[0], lon[0])
        xy = np.stack([x, yy], axis=1)
        steps = np.diff(xy, axis=0, prepend=xy[:1])  # first step = 0

        # Fill missing speed/course from the track itself.
        speed = np.linalg.norm(steps, axis=1) / step_s / KN_TO_MS
        sog = seg["sog"].fillna(pd.Series(speed, index=seg.index)).to_numpy()
        heading = np.arctan2(steps[:, 0], steps[:, 1])
        cs = seg["cog_sin"].fillna(pd.Series(np.sin(heading), index=seg.index)).to_numpy()
        cc = seg["cog_cos"].fillna(pd.Series(np.cos(heading), index=seg.index)).to_numpy()

        feats = np.stack([steps[:, 0], steps[:, 1], sog, cs, cc], axis=1)
        for i in range(1, len(seg) - WINDOW):
            j = i + WINDOW  # index of the point to predict
            X.append(feats[i:j])
            y.append(xy[j] - xy[j - 1])
            last_xy.append(xy[j - 2 : j])
            # Dead reckoning: reported speed and course at the last known point.
            dist = sog[j - 1] * KN_TO_MS * step_s
            dr.append([dist * cs[j - 1], dist * cc[j - 1]])
            mmsis.append(seg["mmsi"].iloc[0])
    return (
        np.asarray(X, np.float32),
        np.asarray(y, np.float32),
        np.asarray(last_xy, np.float32),
        np.asarray(dr, np.float32),
        np.asarray(mmsis),
    )


def errors_m(pred_disp: np.ndarray, true_disp: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred_disp - true_disp, axis=1)


def summarize(name, err, freq, n):
    return {
        "freq": freq,
        "method": name,
        "mean_error_m": float(err.mean()),
        "median_error_m": float(np.median(err)),
        "p90_error_m": float(np.percentile(err, 90)),
        "n_test": n,
    }


def run_freq(df: pd.DataFrame, freq: str, rng: np.random.Generator):
    step_s = pd.Timedelta(freq).total_seconds()
    max_gap = max(pd.Timedelta("10min"), 3 * pd.Timedelta(freq))
    segments = data.segment_tracks(df, freq, max_gap=max_gap, min_points=WINDOW + 5)
    X, y, last_xy, dr, mmsi = build_samples(segments, step_s)

    vessels = np.unique(mmsi)
    rng.shuffle(vessels)
    n = len(vessels)
    test_v = set(vessels[: int(0.15 * n)])
    val_v = set(vessels[int(0.15 * n) : int(0.30 * n)])
    te = np.isin(mmsi, list(test_v))
    va = np.isin(mmsi, list(val_v))
    tr = ~(te | va)
    print(
        f"[{freq}] {n} vessels, samples train/val/test = "
        f"{tr.sum()}/{va.sum()}/{te.sum()}"
    )

    # Standardise inputs with training-set statistics only. The dead-reckoning
    # step is added as an extra input so the hybrid model knows what it corrects.
    X = np.concatenate([X, np.repeat(dr[:, None, :], X.shape[1], axis=1)], axis=2)
    x_mu = X[tr].reshape(-1, X.shape[2]).mean(0)
    x_sd = X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
    Xn = (X - x_mu) / x_sd

    def train(target: np.ndarray, name: str):
        """Train an LSTM on a standardised target; return test predictions in meters."""
        t_mu, t_sd = target[tr].mean(0), target[tr].std(0) + 1e-6
        tn = (target - t_mu) / t_sd
        models.set_seed(42)
        model = models.displacement_lstm(WINDOW, X.shape[2])
        hist = model.fit(
            Xn[tr],
            tn[tr],
            validation_data=(Xn[va], tn[va]),
            epochs=100,
            batch_size=512,
            verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
            ],
        )
        model.save(data.ROOT / "models" / f"{name}_{freq}.keras")
        pred = model.predict(Xn[te], batch_size=4096, verbose=0) * t_sd + t_mu
        return pred, hist.history

    # 1) Pure LSTM: predicts the next displacement directly.
    pred_lstm, hist_lstm = train(y, "lstm")
    # 2) Hybrid: physics first, LSTM learns only the correction to dead reckoning.
    resid, hist_hyb = train(y - dr, "hybrid")
    pred_hybrid = dr[te] + resid

    true = y[te]
    lx = last_xy[te]
    n_te = int(te.sum())
    rows = [
        summarize("Hybrid: dead reckoning + LSTM", errors_m(pred_hybrid, true), freq, n_te),
        summarize("LSTM (displacement)", errors_m(pred_lstm, true), freq, n_te),
        summarize("Dead reckoning (SOG/COG)", errors_m(dr[te], true), freq, n_te),
        summarize("Constant velocity", errors_m(lx[:, 1] - lx[:, 0], true), freq, n_te),
        summarize("Persistence (stay put)", errors_m(np.zeros_like(true), true), freq, n_te),
    ]
    extra = {
        "freq": freq,
        "n_vessels": n,
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "history_lstm": hist_lstm,
        "history_hybrid": hist_hyb,
        "mean_step_m": float(np.linalg.norm(true, axis=1).mean()),
    }
    return rows, extra


def main():
    print(models.gpu_summary())
    (data.ROOT / "models").mkdir(exist_ok=True)
    df = data.load_clean()
    print(f"{len(df):,} clean Class A position reports, {df['mmsi'].nunique()} vessels")

    rows, extras = [], []
    for freq in FREQS:
        r, e = run_freq(df, freq, np.random.default_rng(42))
        rows += r
        extras.append(e)
        print(pd.DataFrame(r).round(1).to_string(index=False))

    out = data.ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "experiment_metrics.csv", index=False)
    with open(out / "experiment_details.json", "w") as f:
        json.dump(extras, f)
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
