# Transformer Model Inference Framework

A modular Python inference framework built on Hugging Face Transformers and PyTorch.
Supports text, vision, audio, and multimodal workloads through a consistent
4-stage pipeline: **load → preprocess → infer → postprocess**.

## Project Structure

```
transformer_inference_framework/
├── core/
│   ├── model_loader.py      # Stage 1: load pretrained model + processor
│   ├── preprocessor.py      # Stage 2: raw input -> tensors
│   ├── inference_engine.py  # Stage 3: forward pass / generate()
│   └── output_handler.py    # Stage 4: tensors -> readable result
├── modalities/
│   ├── text_pipeline.py
│   ├── vision_pipeline.py
│   ├── audio_pipeline.py
│   └── multimodal_pipeline.py
├── pipeline.py               # single entrypoint: run(model_name, modality, data)
├── scripts/                  # runnable demos per modality
├── tests/                    # unit tests per core component
└── api/main.py                # FastAPI wrapper over pipeline.run()
```

## Setup

```bash
cd transformer_inference_framework
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Text classification demo:
```bash
python scripts/run_text_demo.py
```

Vision classification demo:
```bash
python scripts/run_vision_demo.py path/to/image.jpg
```

Audio classification demo:
```bash
python scripts/run_audio_demo.py path/to/audio.wav
```

Multimodal (image + text matching) demo:
```bash
python scripts/run_multimodal_demo.py path/to/image.jpg
```

Programmatic usage:
```python
from pipeline import run

result = run(
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    modality="text",
    data="This framework is great!",
)
print(result)
```

## Which Hugging Face models will actually work here

`model_name` is a parameter, not a hardcoded value — the 4 model names used in the demo
scripts are just convenient defaults. Any model on the Hub can be used **as long as it
satisfies all 4 of the following**:

1. **It matches one of the 6 supported task types.** `core/model_loader.py` maps each
   `task` to a specific Hugging Face auto-class:

   | `task` value | Model class used | What the model must be |
   |---|---|---|
   | `text-classification` | `AutoModelForSequenceClassification` | A text model with a classification head (sentiment, spam detection, etc.) |
   | `text-generation` | `AutoModelForCausalLM` | A causal/autoregressive text model (GPT-style) |
   | `text-seq2seq` | `AutoModelForSeq2SeqLM` | A translation/summarization-style model (T5, BART) |
   | `vision` | `AutoModelForImageClassification` | An image classification model |
   | `audio` | `AutoModelForAudioClassification` | An audio classification model |
   | `multimodal` | `AutoModel` | Specifically a **CLIP-style** model (image-text similarity) |

2. **It exists, is spelled correctly, and is public.** Gated/private repos (e.g. Meta's
   Llama family) will fail — no auth token is passed by the loader.
3. **It ships a matching processor.** The loader always fetches a tokenizer, image
   processor, feature extractor, or processor alongside the model — repos missing that
   config will fail even if the model weights exist.
4. **Its output shape matches what `output_handler.py` expects.** This is narrowest for
   `multimodal`: the output decoder specifically expects a CLIP-style `.logits_per_image`
   attribute, so non-CLIP multimodal architectures will not decode correctly even if they load.

If a model name doesn't resolve on the Hub, `ModelLoader.load()` raises a clean
`ModelLoadError` (instead of crashing) with a message explaining the likely cause; the API
layer catches this and returns `HTTP 400` with that message instead of a generic 500.

### Examples that work out of the box

| Task | Example models |
|---|---|
| `text-classification` | `distilbert-base-uncased-finetuned-sst-2-english`, `cardiffnlp/twitter-roberta-base-sentiment-latest`, `nlptown/bert-base-multilingual-uncased-sentiment` |
| `text-generation` | `gpt2`, `distilgpt2` |
| `text-seq2seq` | `t5-small`, `facebook/bart-large-cnn` |
| `vision` | `google/vit-base-patch16-224`, `microsoft/resnet-50` |
| `audio` | `superb/wav2vec2-base-superb-ks`, `superb/hubert-base-superb-ks` |
| `multimodal` | `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14` |

### Examples that will NOT work with this codebase

| Model type | Example | Why it fails |
|---|---|---|
| Image generation (diffusion) | Stable Diffusion | Not one of the 6 supported task types |
| Speech-to-text transcription | OpenAI Whisper | Audio→text (seq2seq), not "audio classification" |
| Modern vision-language chat models | LLaVA, Qwen-VL | Output shape isn't CLIP-style, breaks the output decoder |
| Sentence embedding models | `sentence-transformers/*` | No classification head |
| Object detection / segmentation | DETR, SAM | Different task, not in the supported list |
| Models needing custom code | Various experimental repos | Requires `trust_remote_code=True`, which isn't passed |

This scope is intentional: the framework covers the 6 classic transformer task types
rather than every model category on the Hub (diffusion, speech synthesis, agentic models,
etc. would each need their own pipeline and output-handling logic).

## Running tests

```bash
pytest tests/ -v
```

## Running the API

```bash
uvicorn api.main:app --reload
```

Interactive docs (with a built-in "Try it out" button) are available at
`http://localhost:8000/docs` once the server is running.

### Text

POST to `http://localhost:8000/infer/text` as JSON:
```json
{
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "text": "This framework is great!"
}
```

### Vision

POST to `http://localhost:8000/infer/vision` as a multipart form (file upload) —
in `/docs` this renders as a file picker button:
```bash
curl -X POST http://localhost:8000/infer/vision \
  -F "model_name=google/vit-base-patch16-224" \
  -F "file=@path/to/image.jpg"
```

### Audio

POST to `http://localhost:8000/infer/audio` the same way, with a wav file:
```bash
curl -X POST http://localhost:8000/infer/audio \
  -F "model_name=superb/wav2vec2-base-superb-ks" \
  -F "file=@path/to/audio.wav"
```

## Verified results (from real, live runs — not simulated)

Every component below was actually executed against a real downloaded model, not just
unit-tested with mocks. Two real bugs were found and fixed this way (a mislabeled text
classification output, and a multimodal pipeline that ran without crashing but returned
no usable result) — see the "Which Hugging Face models will actually work here" section
above for the constraints these tests confirmed.

| Check | Command | Result |
|---|---|---|
| Unit tests | `pytest tests/ -v` | 10/10 passed |
| Text (script) | `python scripts/run_text_demo.py` | `predicted_label: POSITIVE`, confidence 0.9996 |
| Vision (script) | `python scripts/run_vision_demo.py <image>` | `predicted_label: tray`, confidence 0.16 (random-noise test image) |
| Audio (script) | `python scripts/run_audio_demo.py <wav>` | `predicted_label: stop`, confidence 0.34 (synthetic tone, not real speech) |
| Multimodal (script) | `python scripts/run_multimodal_demo.py <image>` | Correctly matched "a photo of random noise" at 99.87% against a random-noise test image |
| Text (API) | `POST /infer/text` | `200 OK`, correct prediction |
| Vision (API) | `POST /infer/vision` | `200 OK`, matches script output |
| Audio (API) | `POST /infer/audio` | `200 OK`, matches script output |
| Invalid model name (API) | `POST /infer/text` with a nonexistent model | `400 Bad Request` with a clear message (previously an unhandled `500`) |
| Different, non-hardcoded model | `run(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", ...)` | Worked with zero code changes — confirms `model_name` isn't hardcoded, just a parameter |
