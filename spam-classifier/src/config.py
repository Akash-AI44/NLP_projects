from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "raw" / "spam.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "processed" / "spam_clean.csv"

RANDOM_STATE = 42
