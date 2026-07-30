"""Demo: run audio classification on a sample WAV file.

Usage:
    python scripts/run_audio_demo.py path/to/audio.wav
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf
from pipeline import run

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_audio_demo.py <path_to_wav>")
        sys.exit(1)

    audio_array, sampling_rate = sf.read(sys.argv[1])
    result = run(
        model_name="superb/wav2vec2-base-superb-ks",
        modality="audio",
        data=audio_array,
        sampling_rate=sampling_rate,
    )
    print(result)
