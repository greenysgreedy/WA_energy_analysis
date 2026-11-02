from pathlib import Path
import re
import time

import requests

from urllib.parse import urljoin

BASE_URL = "https://data.wa.aemo.com.au/public/public-data/datafiles"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

DATASETS = {
    "reference-trading-price-csv": None,
    "market-clearing-prices-csv": None,
    "fcess": None,
    "capacity-credits-since-market-start": None,  
    "facility-scada-csv": None,
}

def list_files(dataset: str) -> list[str]:
    """Return the file links on a dataset's index page."""
    resp = requests.get(f"{BASE_URL}/{dataset}/", timeout=30)
    resp.raise_for_status()

    links = re.findall(r'href="([^"]+)"', resp.text, flags=re.IGNORECASE)

    folder = f"{BASE_URL}/{dataset}/"
    return [urljoin(folder, l) for l in links
            if l.lower().endswith(".csv")]

def download_files(dataset: str, file: str) -> bool:
    """Download a file from a dataset's index page."""
    name = file.split("/")[-1]
    out = RAW_DIR / dataset / name
    if out.exists():
        print(f"skip {name} - it already exists")
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    url = file if file.startswith("http") else f"{file}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    print(f"saved {name} to {out}")
    return True

if __name__ == "__main__":
    for dataset_name, dataset_config in list(DATASETS.items())[-1:]:
        print(f"Listing files for {dataset_name}")
        files = list_files(dataset_name)
        for file in files:
            download_files(dataset_name, file)
            time.sleep(6.7) 



