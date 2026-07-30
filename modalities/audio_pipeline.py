"""Audio modality: wires up the core stages for audio classification."""

from core.model_loader import ModelLoader
from core.preprocessor import Preprocessor
from core.inference_engine import InferenceEngine
from core.output_handler import OutputHandler


class AudioPipeline:
    def __init__(self, model_name: str):
        task = "audio"
        loader = ModelLoader(model_name, task)
        model, feature_extractor = loader.load()

        self.task = task
        self.preprocessor = Preprocessor(feature_extractor, task, loader.device)
        self.engine = InferenceEngine(model, task)
        self.output_handler = OutputHandler(model.config, task)

    def run(self, audio_array, sampling_rate: int = 16000) -> dict:
        """audio_array: a 1D numpy array of raw waveform samples."""
        inputs = self.preprocessor._preprocess_audio(audio_array, sampling_rate)
        raw_output = self.engine.run(inputs)
        return self.output_handler.run(raw_output)
