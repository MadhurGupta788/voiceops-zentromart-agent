import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / os.getenv("DATASET_DIR", "data_local/Capstone_Project_Dataset_FTTT_2026")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
TODAY = "20 October 2026"
RESULTS_DIR = ROOT / "results"
