from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"

RUNS_DIR = ROOT / "runs"
FIGURES_DIR = ROOT / "figures"

for p in [DATA_DIR, RAW_DATA, PROCESSED_DATA, RUNS_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)
