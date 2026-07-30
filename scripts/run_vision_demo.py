"""Demo: run image classification on a sample image.

Usage:
    python scripts/run_vision_demo.py path/to/image.jpg
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from pipeline import run

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_vision_demo.py <path_to_image>")
        sys.exit(1)

    image = Image.open(sys.argv[1]).convert("RGB")
    result = run(
        model_name="google/vit-base-patch16-224",
        modality="vision",
        data=image,
    )
    print(result)
