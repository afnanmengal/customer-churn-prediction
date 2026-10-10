"""Download the Telco Customer Churn CSV from Kaggle."""

import argparse
import shutil
import tempfile
from pathlib import Path

import kagglehub

DATASET = "blastchar/telco-customer-churn"
FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Download the expected CSV into the ignored data/raw directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="telco-churn-") as temporary:
        download = Path(kagglehub.dataset_download(DATASET, output_dir=temporary))
        source = next((path for path in download.rglob(FILENAME) if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"{FILENAME} not found in Kaggle download")
        destination = output_dir / FILENAME
        shutil.copyfile(source, destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
