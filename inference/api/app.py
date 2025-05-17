"""
DeepShiva FastAPI Web Service

This module provides a REST API for the DeepShiva model, allowing for:
- Text generation
- Code completion
- Mathematical reasoning
- Translation between languages
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Union

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import DeepShiva inference module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference import DeepShivaInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DeepShiva API",
    description="API for DeepShiva MoE model for code generation, mathematical reasoning, and multilingual support",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for inference engine
inference_engine = None


# Pydantic models for request/response
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, description="Penalty for repeating tokens")
    do_sample: bool = Field(True, description="Whether to use sampling or greedy decoding")
    num_return_sequences: int = Field(1, description="Number of sequences to return")
    stop_strings: Optional[List[str]] = Field(None, description="List of strings to stop generation when encountered")


class CodeCompletionRequest(BaseModel):
    code_prompt: str = Field(..., description="Code prompt to complete")
    language: str = Field("python", description="Programming language")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.2, description="Sampling temperature")
    top_p: float = Field(0.95, description="Nucleus sampling parameter")
    repetition_penalty: float = Field(1.05, description="Penalty for repeating tokens")


class MathSolvingRequest(BaseModel):
    problem: str = Field(..., description="Math problem to solve")
    max_new_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.3, description="Sampling temperature")
    show_work: bool = Field(True, description="Whether to show the work or just the answer")


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_lang: str = Field(..., description="Source language code (e.g., 'en', 'hi')")
    target_lang: str = Field(..., description="Target language code (e.g., 'en', 'hi')")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.5, description="Sampling temperature")


class GenerateResponse(BaseModel):
    generated_text: Union[str, List[str]] = Field(..., description="Generated text")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: Optional[str] = Field(None, description="Path to the loaded model")


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global inference_engine
    
    # Get model path from environment variable or use default
    model_path = os.environ.get("DEEPSHIVA_MODEL_PATH", "models/pretrained/deepshiva-moe")
    device = os.environ.get("DEEPSHIVA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    precision = os.environ.get("DEEPSHIVA_PRECISION", "fp16")
    
    try:
        logger.info(f"Loading model from {model_path}")
        inference_engine = DeepShivaInference(
            model_path=model_path,
            device=device,
            precision=precision,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Continue without model, will return error on API calls


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API."""
    global inference_engine
    
    if inference_engine is None:
        return HealthResponse(
            status="warning",
            model_loaded=False,
            model_path=None,
        )
    
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_path=inference_engine.model_path,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text based on the prompt."""
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated_text = inference_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            num_return_sequences=request.num_return_sequences,
            stop_strings=request.stop_strings,
        )
        
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/code", response_model=GenerateResponse)
async def complete_code(request: CodeCompletionRequest):
    """Complete code based on the prompt."""
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        completed_code = inference_engine.complete_code(
            code_prompt=request.code_prompt,
            language=request.language,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        
        return GenerateResponse(generated_text=completed_code)
    except Exception as e:
        logger.error(f"Error completing code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/math", response_model=GenerateResponse)
async def solve_math(request: MathSolvingRequest):
    """Solve a mathematical problem."""
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        solution = inference_engine.solve_math(
            problem=request.problem,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            show_work=request.show_work,
        )
        
        return GenerateResponse(generated_text=solution)
    except Exception as e:
        logger.error(f"Error solving math problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate", response_model=GenerateResponse)
async def translate(request: TranslationRequest):
    """Translate text from source language to target language."""
    global inference_engine
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        translation = inference_engine.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        
        return GenerateResponse(generated_text=translation)
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("DEEPSHIVA_API_PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
