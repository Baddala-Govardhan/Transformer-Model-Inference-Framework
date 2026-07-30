"""Top-level entrypoint: dispatches a request to the right modality pipeline."""

from modalities.text_pipeline import TextPipeline
from modalities.vision_pipeline import VisionPipeline
from modalities.audio_pipeline import AudioPipeline
from modalities.multimodal_pipeline import MultimodalPipeline

_PIPELINE_CACHE: dict[tuple[str, str], object] = {}


def _get_pipeline(model_name: str, modality: str, task: str | None):
    key = (model_name, modality)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    if modality == "text":
        pipeline = TextPipeline(model_name, task=task or "text-classification")
    elif modality == "vision":
        pipeline = VisionPipeline(model_name)
    elif modality == "audio":
        pipeline = AudioPipeline(model_name)
    elif modality == "multimodal":
        pipeline = MultimodalPipeline(model_name)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    _PIPELINE_CACHE[key] = pipeline
    return pipeline


def run(
    model_name: str,
    modality: str,
    data,
    task: str | None = None,
    sampling_rate: int = 16000,
) -> dict:
    """
    Single entrypoint used by scripts, tests, and the API layer.

    model_name:    Hugging Face model id, e.g. "distilbert-base-uncased-finetuned-sst-2-english"
    modality:      one of "text", "vision", "audio", "multimodal"
    data:          raw input (str for text, PIL.Image for vision, np.ndarray for audio,
                   dict {"text":..., "image":...} for multimodal)
    task:          optional sub-task override for text (e.g. "text-generation")
    sampling_rate: the *actual* sample rate of the audio in `data` (ignored for
                   other modalities) - the pipeline downmixes/resamples to 16kHz
                   internally, but it needs to know the real rate to do that correctly
    """
    pipeline = _get_pipeline(model_name, modality, task)

    if modality == "multimodal":
        result = pipeline.run(text=data["text"], image=data["image"])
    elif modality == "audio":
        result = pipeline.run(data, sampling_rate=sampling_rate)
    else:
        result = pipeline.run(data)

    result["modality"] = modality
    result["model"] = model_name
    return result
