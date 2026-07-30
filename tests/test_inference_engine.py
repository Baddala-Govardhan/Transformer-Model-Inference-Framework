import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from core.inference_engine import InferenceEngine


class FakeClassificationModel:
    class Output:
        def __init__(self, logits):
            self.logits = logits

    def __call__(self, **kwargs):
        return self.Output(logits=torch.tensor([[0.1, 0.9]]))


class FakeGenerativeModel:
    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])


def test_classification_forward_pass():
    engine = InferenceEngine(FakeClassificationModel(), task="text-classification")
    output = engine.run({"input_ids": torch.tensor([[1, 2, 3]])})
    assert output.logits.shape == (1, 2)


def test_generation_calls_generate():
    engine = InferenceEngine(FakeGenerativeModel(), task="text-generation")
    output = engine.run({"input_ids": torch.tensor([[1, 2, 3]])})
    assert output.tolist() == [[1, 2, 3]]
