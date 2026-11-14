"""Download one day of AIS data from the Danish Maritime Authority (DMA).

The DMA publishes raw AIS messages for Danish waters free of charge at
http://aisdata.ais.dk/. Each daily file is a ~500 MB zip with a ~3 GB CSV.

Usage:
    python scripts/download_data.py              # the day used in the thesis
    python scripts/download_data.py 2025-03-01   # any other day
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_DAY = "2025-02-21"


def download(day: str = DEFAULT_DAY) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    name = f"aisdk-{day}"
    csv_path = RAW_DIR / f"{name}.csv"
    if csv_path.exists():
        print(f"Already have {csv_path}")
        return csv_path

    zip_path = RAW_DIR / f"{name}.zip"
    if not zip_path.exists():
        url = f"http://aisdata.ais.dk/{day[:4]}/{name}.zip"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    return csv_path


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DAY)
