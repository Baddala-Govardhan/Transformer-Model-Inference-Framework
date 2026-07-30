"""Demo: run a CLIP-style multimodal model on a sample image + candidate text.

Usage:
    python scripts/run_multimodal_demo.py path/to/image.jpg
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from pipeline import run

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_multimodal_demo.py <path_to_image>")
        sys.exit(1)

    image = Image.open(sys.argv[1]).convert("RGB")
    result = run(
        model_name="openai/clip-vit-base-patch32",
        modality="multimodal",
        data={
            "text": ["a photo of random noise", "a photo of a cat", "a photo of a mountain"],
            "image": image,
        },
    )
    print(result)
