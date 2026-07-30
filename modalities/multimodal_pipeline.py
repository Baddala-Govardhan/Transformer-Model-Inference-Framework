"""Multimodal: wires up the core stages for combined text+image (e.g. CLIP-style) models."""

from core.model_loader import ModelLoader
from core.preprocessor import Preprocessor
from core.inference_engine import InferenceEngine
from core.output_handler import OutputHandler


class MultimodalPipeline:
    def __init__(self, model_name: str):
        task = "multimodal"
        loader = ModelLoader(model_name, task)
        model, processor = loader.load()

        self.task = task
        self.preprocessor = Preprocessor(processor, task, loader.device)
        self.engine = InferenceEngine(model, task)
        self.output_handler = OutputHandler(processor, task)

    def run(self, text, image) -> dict:
        """text: a single candidate label or a list of candidate labels to compare
        against the image (CLIP-style zero-shot classification)."""
        candidates = text if isinstance(text, list) else [text]
        inputs = self.preprocessor.run({"text": candidates, "images": image})
        raw_output = self.engine.run(inputs)
        return self.output_handler.run(raw_output, candidates=candidates)
