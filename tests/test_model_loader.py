import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch

import pytest
from core.model_loader import ModelLoader, ModelLoadError, get_device


def test_get_device_returns_valid_string():
    assert get_device() in ("cpu", "cuda", "mps")


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        ModelLoader("some-model", task="not-a-real-task")


def test_loader_stores_config():
    loader = ModelLoader("distilbert-base-uncased-finetuned-sst-2-english", task="text-classification")
    assert loader.model_name == "distilbert-base-uncased-finetuned-sst-2-english"
    assert loader.task == "text-classification"
    assert loader.model is None
    assert loader.processor is None


def test_invalid_model_name_raises_clean_error():
    loader = ModelLoader("this-model-does-not-exist-12345", task="text-classification")
    with patch(
        "core.model_loader.AutoModelForSequenceClassification.from_pretrained",
        side_effect=OSError("not found on hub"),
    ):
        with pytest.raises(ModelLoadError):
            loader.load()
