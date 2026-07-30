"""Stage 2: Convert raw input into model-ready tensors, dispatched by modality."""

from typing import Any

import numpy as np

TARGET_SAMPLE_RATE = 16000


class Preprocessor:
    def __init__(self, processor, task: str, device: str):
        self.processor = processor
        self.task = task
        self.device = device

    def run(self, raw_input: Any):
        if self.task.startswith("text"):
            return self._preprocess_text(raw_input)
        if self.task == "vision":
            return self._preprocess_vision(raw_input)
        if self.task == "audio":
            return self._preprocess_audio(raw_input)
        if self.task == "multimodal":
            return self._preprocess_multimodal(raw_input)
        raise ValueError(f"Unsupported task: {self.task}")

    def _preprocess_text(self, text: str):
        inputs = self.processor(text, return_tensors="pt", truncation=True, padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _preprocess_vision(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _preprocess_audio(self, audio_array, sampling_rate: int = TARGET_SAMPLE_RATE):
        audio_array, sampling_rate = self._normalize_audio(audio_array, sampling_rate)
        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    @staticmethod
    def _normalize_audio(audio_array, sampling_rate: int):
        """Downmix to mono and resample to 16kHz - the format wav2vec2-style
        audio models require. Uploaded files (e.g. stereo mp3s at 44.1kHz)
        won't match this by default, so this always runs before feature extraction."""
        audio_array = np.asarray(audio_array, dtype=np.float32)

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        if sampling_rate != TARGET_SAMPLE_RATE:
            duration = len(audio_array) / sampling_rate
            target_length = max(1, int(duration * TARGET_SAMPLE_RATE))
            audio_array = np.interp(
                np.linspace(0, len(audio_array), target_length, endpoint=False),
                np.arange(len(audio_array)),
                audio_array,
            ).astype(np.float32)
            sampling_rate = TARGET_SAMPLE_RATE

        return audio_array, sampling_rate

    def _preprocess_multimodal(self, raw_input: dict):
        # raw_input expected as {"text": [...], "images": ...}
        inputs = self.processor(**raw_input, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items()}
