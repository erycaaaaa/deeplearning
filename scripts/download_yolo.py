"""Run once with: python scripts/download_yolo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.detector import ensure_weights

if __name__ == "__main__":
    print(f"YOLO weights ready: {ensure_weights()}")
