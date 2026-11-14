"""Figures for the README, built from the saved results files.

Run:  python -m ais_traj.plots
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data

RES = data.ROOT / "results"
FIG = data.ROOT / "figures"
COLORS = {
    "Thesis LSTM": "#2563eb",
    "LSTM (displacement)": "#2563eb",
    "Persistence (stay put)": "#9ca3af",
    "Constant velocity": "#f59e0b",
    "Dead reckoning (SOG/COG)": "#10b981",
}


def bar_chart(df: pd.DataFrame, title: str, path):
    freqs = list(dict.fromkeys(df["freq"]))
    methods = list(dict.fromkeys(df["method"]))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, m in enumerate(methods):
        vals = [df[(df.freq == f) & (df.method == m)]["median_error_m"].iloc[0] for f in freqs]
        xs = np.arange(len(freqs)) + (i - (len(methods) - 1) / 2) * w
        bars = ax.bar(xs, vals, w, label=m, color=COLORS.get(m))
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(range(len(freqs)), [f"{f} ahead" for f in freqs])
    ax.set_ylabel("Median position error (m)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def thesis_track_plot(details: dict, path):
    run = details["runs"][0]  # 1-minute run
    track = np.array(run["track"])
    true = np.array(run["test_true"])
    pred = np.array(run["test_pred"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(track[:, 1], track[:, 0], color="#d1d5db", lw=1, label="Full day track")
    ax.plot(true[:, 1], true[:, 0], color="#111827", lw=1.5, label="Test: actual")
    ax.plot(pred[:, 1], pred[:, 0], color="#2563eb", lw=1, ls="--", label="Test: LSTM")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Thesis vessel (MMSI {details['mmsi']}), 1-minute steps")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    if (RES / "reproduce_metrics.csv").exists():
        bar_chart(
            pd.read_csv(RES / "reproduce_metrics.csv"),
            "Thesis setup: one vessel",
            FIG / "reproduce_errors.png",
        )
        with open(RES / "reproduce_details.json") as f:
            thesis_track_plot(json.load(f), FIG / "thesis_vessel_track.png")
    if (RES / "experiment_metrics.csv").exists():
        bar_chart(
            pd.read_csv(RES / "experiment_metrics.csv"),
            "Improved setup: all moving vessels, unseen test ships",
            FIG / "experiment_errors.png",
        )
    print(f"Figures written to {FIG}")


if __name__ == "__main__":
    main()
