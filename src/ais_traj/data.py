"""Loading, cleaning and segmenting raw DMA AIS data."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

COLUMNS = {
    "# Timestamp": "timestamp",
    "Type of mobile": "mobile_type",
    "MMSI": "mmsi",
    "Latitude": "lat",
    "Longitude": "lon",
    "Navigational status": "nav_status",
    "SOG": "sog",
    "COG": "cog",
    "Ship type": "ship_type",
}

# Rough bounding box around Danish waters; drops the obviously bad fixes
# (e.g. lat 91 / lon 181 "not available" values and a few spoofed positions).
LAT_RANGE = (53.0, 59.5)
LON_RANGE = (2.0, 17.0)

EARTH_RADIUS_M = 6_371_000.0


def load_raw(day: str = "2025-02-21") -> pd.DataFrame:
    """Load the full raw CSV (all columns), as the thesis notebook did."""
    return pd.read_csv(RAW_DIR / f"aisdk-{day}.csv", engine="pyarrow")


def load_clean(day: str = "2025-02-21") -> pd.DataFrame:
    """Load Class A position reports with valid coordinates, cached as parquet."""
    cache = PROCESSED_DIR / f"aisdk-{day}-clean.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    df = pd.read_csv(
        RAW_DIR / f"aisdk-{day}.csv",
        usecols=list(COLUMNS),
        engine="pyarrow",
    ).rename(columns=COLUMNS)

    df = df[df["mobile_type"] == "Class A"]
    df = df[df["lat"].between(*LAT_RANGE) & df["lon"].between(*LON_RANGE)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M:%S")
    df = (
        df.drop(columns="mobile_type")
        .drop_duplicates(["mmsi", "timestamp"])
        .sort_values(["mmsi", "timestamp"])
        .reset_index(drop=True)
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def segment_tracks(
    df: pd.DataFrame,
    freq: str,
    max_gap: str = "10min",
    max_speed_kn: float = 50.0,
    min_points: int = 30,
) -> list[pd.DataFrame]:
    """Split each vessel's track at reporting gaps and resample each piece.

    Unlike the thesis notebook, we never interpolate across a long gap: a
    vessel that went silent for an hour starts a new segment instead of
    getting an invented straight-line path.
    """
    gap = pd.Timedelta(max_gap)
    segments = []
    for mmsi, track in df.groupby("mmsi", sort=False):
        track = track.set_index("timestamp")
        dt = track.index.to_series().diff()
        # Drop physically impossible jumps (GPS glitches).
        dist = haversine_m(
            track["lat"].shift(), track["lon"].shift(), track["lat"], track["lon"]
        )
        speed_kn = dist / dt.dt.total_seconds().clip(lower=1) / 0.514444
        track = track[~(speed_kn > max_speed_kn)]
        dt = track.index.to_series().diff()

        seg_id = (dt > gap).cumsum()
        for _, seg in track.groupby(seg_id.values):
            # Course is circular (359° is next to 0°), so average it as a unit vector.
            num = seg[["lat", "lon", "sog"]].assign(
                cog_sin=np.sin(np.radians(seg["cog"])),
                cog_cos=np.cos(np.radians(seg["cog"])),
            )
            res = num.resample(freq).mean().interpolate("linear", limit_area="inside")
            res = res.dropna(subset=["lat", "lon"])
            if len(res) >= min_points:
                res["mmsi"] = mmsi
                res["ship_type"] = seg["ship_type"].iloc[0]
                segments.append(res)
    return segments
