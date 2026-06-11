"""
Download the OPSD Time Series dataset (60-minute resolution).
Source: https://data.open-power-system-data.org/time_series/2020-10-06
"""

import os
import requests
import sys


DATA_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "time_series_60min_singleindex.csv")


def download_dataset():
    """Download the OPSD 60-min time series CSV."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"[INFO] Dataset already exists: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
        return OUTPUT_FILE

    print(f"[INFO] Downloading dataset from OPSD...")
    print(f"[INFO] URL: {DATA_URL}")

    response = requests.get(DATA_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(OUTPUT_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r[DOWNLOAD] {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="")

    print()
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"[SUCCESS] Dataset downloaded: {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    return OUTPUT_FILE


if __name__ == "__main__":
    download_dataset()
