import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from core.preprocessor import Preprocessor


class FakeTextProcessor:
    def __call__(self, text, return_tensors=None, truncation=None, padding=None):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def test_text_preprocessing_moves_to_device():
    processor = Preprocessor(FakeTextProcessor(), task="text-classification", device="cpu")
    result = processor.run("hello world")
    assert "input_ids" in result
    assert result["input_ids"].device.type == "cpu"


def test_unsupported_task_raises():
    processor = Preprocessor(FakeTextProcessor(), task="not-a-task", device="cpu")
    try:
        processor.run("hi")
        assert False, "expected ValueError"
    except ValueError:
        pass
