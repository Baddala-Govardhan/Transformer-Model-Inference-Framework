import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from core.output_handler import OutputHandler


class FakeClassificationOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeProcessorWithLabels:
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}


class FakeTokenizer:
    def batch_decode(self, ids, skip_special_tokens=True):
        return ["decoded text"]


def test_classification_output_returns_label_and_confidence():
    handler = OutputHandler(FakeProcessorWithLabels(), task="text-classification")
    raw_output = FakeClassificationOutput(logits=torch.tensor([[0.1, 0.9]]))
    result = handler.run(raw_output)
    assert result["predicted_label"] == "POSITIVE"
    assert 0.0 <= result["confidence"] <= 1.0


def test_generation_output_decodes_text():
    handler = OutputHandler(FakeTokenizer(), task="text-generation")
    result = handler.run(torch.tensor([[1, 2, 3]]))
    assert result["output"] == ["decoded text"]
