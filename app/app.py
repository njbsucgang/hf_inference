from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline
from typing import Dict, Optional, Any, Union
from PIL import Image
import librosa
import numpy as np
import io
import torch
import json

app = FastAPI()
model_cache: Dict[str, Any] = {}

class InferenceRequest(BaseModel):
    model_name: str
    task: Optional[str] = None
    inputs: Optional[Union[str, Dict[str, Any]]] = None
    parameters: Optional[Dict[str, Any]] = None

async def get_pipeline(model_name: str, task: Optional[str] = None):
    cache_key = f"{model_name}_{task}" if task else model_name
    if cache_key not in model_cache:
        try:
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline(task=task, model=model_name, device=device)
            model_cache[cache_key] = pipe
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model '{model_name}': {str(e)}"
            )
    return model_cache[cache_key]

@app.get("/health")
async def health_check():
    return {"status": "OK"}

@app.post("/infer")
async def infer(
    model_name: str = Form(...),
    task: Optional[str] = Form(None),
    inputs: Optional[str] = Form(None),
    parameters: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        if not inputs and not file:
            raise HTTPException(status_code=400, detail="Either 'inputs' or 'file' must be provided")

        input_data: Union[str, Dict[str, Any], Image.Image] = None

        # Parse JSON string input
        if inputs:
            try:
                input_data = json.loads(inputs)
            except json.JSONDecodeError:
                input_data = inputs  # Fallback to raw string

        # Parse parameters
        inference_params: Dict[str, Any] = {}
        if parameters and parameters.strip():
            try:
                inference_params = json.loads(parameters)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid parameters JSON: {str(e)}")

        # Load model
        pipe = await get_pipeline(model_name, task)

        # Process uploaded file if present
        if file:
            file_content = await file.read()
            file_obj = io.BytesIO(file_content)

            if file.content_type.startswith("image/"):
                input_data = Image.open(file_obj).convert("RGB")
            elif file.content_type.startswith("audio/"):
                audio, sr = librosa.load(file_obj, sr=16000)
                input_data = {"array": audio, "sampling_rate": sr}
            elif file.content_type.startswith("video/"):
                raise HTTPException(status_code=400, detail="Video processing not supported")
            else:
                input_data = file_content  # raw bytes

        # Format text input based on task
        if isinstance(input_data, str):
            if pipe.task in ["text-generation", "text-classification", "fill-mask"]:
                input_data = [input_data]

        # Inference
        try:
            result = pipe(input_data, **inference_params) if inference_params else pipe(input_data)
            return {
                "model": model_name,
                "task": pipe.task,
                "result": result
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Model inference error: {str(e)}")

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Server error: {str(e)}"},
            media_type="application/json"
        )

@app.get("/supported_tasks")
async def list_supported_tasks():
    from transformers.pipelines import SUPPORTED_TASKS
    return {
        "supported_tasks": list(SUPPORTED_TASKS.keys()),
        "message": "Note: Custom models may support additional tasks"
    }
