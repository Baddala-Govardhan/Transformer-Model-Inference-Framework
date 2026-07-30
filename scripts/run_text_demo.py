"""Demo: run sentiment classification on a sample sentence.

Usage:
    python scripts/run_text_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import run

if __name__ == "__main__":
    result = run(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        modality="text",
        task="text-classification",
        data="I really enjoyed building this inference framework!",
    )
    print(result)
