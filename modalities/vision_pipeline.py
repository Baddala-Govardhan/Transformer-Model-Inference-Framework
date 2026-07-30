"""Vision modality: wires up the core stages for image classification."""

from core.model_loader import ModelLoader
from core.preprocessor import Preprocessor
from core.inference_engine import InferenceEngine
from core.output_handler import OutputHandler


class VisionPipeline:
    def __init__(self, model_name: str):
        task = "vision"
        loader = ModelLoader(model_name, task)
        model, image_processor = loader.load()

        self.task = task
        self.preprocessor = Preprocessor(image_processor, task, loader.device)
        self.engine = InferenceEngine(model, task)
        self.output_handler = OutputHandler(model.config, task)

    def run(self, image) -> dict:
        """image: a PIL.Image or a path/array accepted by the HF image processor."""
        inputs = self.preprocessor.run(image)
        raw_output = self.engine.run(inputs)
        return self.output_handler.run(raw_output)
