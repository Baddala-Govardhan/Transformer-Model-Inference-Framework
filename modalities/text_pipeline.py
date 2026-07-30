"""Text modality: wires up the core stages for text classification or generation."""

from core.model_loader import ModelLoader
from core.preprocessor import Preprocessor
from core.inference_engine import InferenceEngine
from core.output_handler import OutputHandler


class TextPipeline:
    def __init__(self, model_name: str, task: str = "text-classification"):
        if not task.startswith("text"):
            raise ValueError("TextPipeline requires a text-* task")

        loader = ModelLoader(model_name, task)
        model, tokenizer = loader.load()

        self.task = task
        self.tokenizer = tokenizer
        self.preprocessor = Preprocessor(tokenizer, task, loader.device)
        self.engine = InferenceEngine(model, task)

        # generation tasks decode via the tokenizer; classification looks up
        # labels from the model config (id2label lives there, not on the tokenizer)
        output_source = tokenizer if task in ("text-generation", "text-seq2seq") else model.config
        self.output_handler = OutputHandler(output_source, task)

    def run(self, text: str, **generate_kwargs) -> dict:
        inputs = self.preprocessor.run(text)
        raw_output = self.engine.run(inputs, **generate_kwargs)
        return self.output_handler.run(raw_output)
