"""Future API integration: a thin FastAPI layer over pipeline.run().

Run with:
    uvicorn api.main:app --reload
"""

import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from core.model_loader import ModelLoadError
from pipeline import run

app = FastAPI(title="Transformer Model Inference Framework")


class TextInferenceRequest(BaseModel):
    model_name: str
    text: str
    task: str = "text-classification"


@app.post("/infer/text")
def infer_text(request: TextInferenceRequest):
    try:
        return run(
            model_name=request.model_name,
            modality="text",
            data=request.text,
            task=request.task,
        )
    except ModelLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


@app.post("/infer/vision")
async def infer_vision(
    model_name: str = Form(default="google/vit-base-patch16-224"),
    file: UploadFile = File(...),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    try:
        return run(model_name=model_name, modality="vision", data=image)
    except ModelLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


@app.post("/infer/audio")
async def infer_audio(
    model_name: str = Form(default="superb/wav2vec2-base-superb-ks"),
    file: UploadFile = File(...),
):
    audio_bytes = await file.read()
    audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
    try:
        return run(
            model_name=model_name,
            modality="audio",
            data=audio_array,
            sampling_rate=sampling_rate,
        )
    except ModelLoadError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}
