"""Stage 1: Load a pretrained model and its matching tokenizer/processor."""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
)

_MODEL_CLASSES = {
    "text-classification": AutoModelForSequenceClassification,
    "text-generation": AutoModelForCausalLM,
    "text-seq2seq": AutoModelForSeq2SeqLM,
    "vision": AutoModelForImageClassification,
    "audio": AutoModelForAudioClassification,
    "multimodal": AutoModel,
}


class ModelLoadError(Exception):
    """Raised when a model name can't be resolved or loaded from the Hub."""


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelLoader:
    """Loads a pretrained model plus the correct processor for a given modality."""

    def __init__(self, model_name: str, task: str, device: str | None = None):
        if task not in _MODEL_CLASSES:
            raise ValueError(f"Unknown task '{task}'. Supported: {list(_MODEL_CLASSES)}")

        self.model_name = model_name
        self.task = task
        self.device = device or get_device()
        self.model = None
        self.processor = None

    def load(self):
        model_cls = _MODEL_CLASSES[self.task]

        try:
            self.model = model_cls.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

            if self.task.startswith("text"):
                self.processor = AutoTokenizer.from_pretrained(self.model_name)
            elif self.task == "vision":
                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            elif self.task == "audio":
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            elif self.task == "multimodal":
                self.processor = AutoProcessor.from_pretrained(self.model_name)
        except OSError as e:
            raise ModelLoadError(
                f"Could not load model '{self.model_name}' for task '{self.task}'. "
                f"Check that the model name is spelled correctly and exists on "
                f"huggingface.co/models, and that it supports this task. ({e})"
            ) from e

        return self.model, self.processor
