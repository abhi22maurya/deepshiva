"""
Enhanced DeepShiva FastAPI Web Service

This module provides an improved REST API for the DeepShiva model with:
- Streaming responses for faster user experience
- Batched processing for higher throughput
- Advanced monitoring and metrics
- Improved error handling
"""

import os
import json
import time
import logging
import asyncio
from typing import List, Dict, Optional, Any, Union, Iterator

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Import DeepShiva inference module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference.enhanced_inference import EnhancedInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DeepShiva Enhanced API",
    description="Enhanced API for DeepShiva MoE model with streaming support and improved performance",
    version="2.0.0",
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

# Metrics tracking
request_count = 0
total_tokens_generated = 0
request_durations = []
request_start_times = {}


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
    stream: bool = Field(False, description="Whether to stream the response")


class BatchGenerateRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of input prompts for generation")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Nucleus sampling parameter")
    top_k: int = Field(50, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, description="Penalty for repeating tokens")
    do_sample: bool = Field(True, description="Whether to use sampling or greedy decoding")


class CodeCompletionRequest(BaseModel):
    code_prompt: str = Field(..., description="Code prompt to complete")
    language: str = Field("python", description="Programming language")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.2, description="Sampling temperature")
    top_p: float = Field(0.95, description="Nucleus sampling parameter")
    repetition_penalty: float = Field(1.05, description="Penalty for repeating tokens")
    stream: bool = Field(False, description="Whether to stream the response")


class MathSolvingRequest(BaseModel):
    problem: str = Field(..., description="Math problem to solve")
    max_new_tokens: int = Field(1024, description="Maximum number of tokens to generate")
    temperature: float = Field(0.3, description="Sampling temperature")
    show_work: bool = Field(True, description="Whether to show the work or just the answer")
    stream: bool = Field(False, description="Whether to stream the response")


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_lang: str = Field(..., description="Source language code (e.g., 'en', 'hi')")
    target_lang: str = Field(..., description="Target language code (e.g., 'en', 'hi')")
    max_new_tokens: int = Field(512, description="Maximum number of tokens to generate")
    temperature: float = Field(0.5, description="Sampling temperature")
    stream: bool = Field(False, description="Whether to stream the response")


class GenerateResponse(BaseModel):
    generated_text: Union[str, List[str]] = Field(..., description="Generated text")
    processing_time: float = Field(..., description="Processing time in seconds")
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")


class BatchGenerateResponse(BaseModel):
    generated_texts: List[str] = Field(..., description="List of generated texts")
    processing_time: float = Field(..., description="Processing time in seconds")
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: Optional[str] = Field(None, description="Path to the loaded model")
    uptime: float = Field(..., description="API uptime in seconds")
    request_count: int = Field(..., description="Number of requests processed")
    avg_processing_time: float = Field(..., description="Average processing time in seconds")
    total_tokens_generated: int = Field(..., description="Total number of tokens generated")


class StreamingToken(BaseModel):
    token: str = Field(..., description="Generated token")
    index: int = Field(..., description="Token index")


# Record start time for uptime calculation
start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global inference_engine
    
    # Get model path from environment variable or use default
    model_path = os.environ.get("DEEPSHIVA_MODEL_PATH", "models/pretrained/deepshiva-moe")
    device = os.environ.get("DEEPSHIVA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    precision = os.environ.get("DEEPSHIVA_PRECISION", "fp16")
    batch_size = int(os.environ.get("DEEPSHIVA_BATCH_SIZE", "1"))
    use_bettertransformer = os.environ.get("DEEPSHIVA_USE_BETTERTRANSFORMER", "true").lower() == "true"
    use_compile = os.environ.get("DEEPSHIVA_USE_COMPILE", "false").lower() == "true"
    
    try:
        logger.info(f"Loading model from {model_path}")
        inference_engine = EnhancedInference(
            model_path=model_path,
            device=device,
            precision=precision,
            batch_size=batch_size,
            use_bettertransformer=use_bettertransformer,
            use_compile=use_compile,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Continue without model, will return error on API calls


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track request processing time and count."""
    global request_count, request_start_times
    
    # Generate request ID
    request_id = str(request_count)
    request_count += 1
    
    # Record start time
    start_time = time.time()
    request_start_times[request_id] = start_time
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    request_durations.append(process_time)
    
    # Add custom header
    response.headers["X-Process-Time"] = str(process_time)
    
    # Clean up
    if request_id in request_start_times:
        del request_start_times[request_id]
    
    return response


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API."""
    global inference_engine, request_count, request_durations, total_tokens_generated, start_time
    
    # Calculate uptime
    uptime = time.time() - start_time
    
    # Calculate average processing time
    avg_processing_time = np.mean(request_durations) if request_durations else 0.0
    
    if inference_engine is None:
        return HealthResponse(
            status="warning",
            model_loaded=False,
            model_path=None,
            uptime=uptime,
            request_count=request_count,
            avg_processing_time=avg_processing_time,
            total_tokens_generated=total_tokens_generated,
        )
    
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_path=inference_engine.model_path,
        uptime=uptime,
        request_count=request_count,
        avg_processing_time=avg_processing_time,
        total_tokens_generated=total_tokens_generated,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text based on the prompt."""
    global inference_engine, total_tokens_generated
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Count input tokens
        input_tokens = len(inference_engine.tokenizer.encode(request.prompt))
        
        # Handle streaming response
        if request.stream:
            return EventSourceResponse(
                _generate_stream_events(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    stop_strings=request.stop_strings,
                ),
                media_type="text/event-stream",
            )
        
        # Standard generation
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
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Count output tokens
        if isinstance(generated_text, list):
            output_tokens = sum(len(inference_engine.tokenizer.encode(text)) for text in generated_text)
        else:
            output_tokens = len(inference_engine.tokenizer.encode(generated_text))
        
        # Update metrics
        total_tokens_generated += output_tokens
        
        return GenerateResponse(
            generated_text=generated_text,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_stream_events(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    do_sample: bool,
    stop_strings: Optional[List[str]],
):
    """Generate streaming events for text generation."""
    global inference_engine, total_tokens_generated
    
    try:
        # Initialize token counter
        token_index = 0
        
        # Generate tokens
        async for token in await asyncio.to_thread(
            inference_engine.generate,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_strings=stop_strings,
            stream=True,
        ):
            # Update metrics
            total_tokens_generated += 1
            
            # Yield token
            yield {
                "event": "token",
                "data": json.dumps({
                    "token": token,
                    "index": token_index,
                }),
            }
            
            # Increment token index
            token_index += 1
            
            # Small delay to avoid overwhelming the client
            await asyncio.sleep(0.01)
        
        # Signal end of stream
        yield {
            "event": "done",
            "data": json.dumps({
                "total_tokens": token_index,
            }),
        }
    except Exception as e:
        logger.error(f"Error in streaming generation: {e}")
        yield {
            "event": "error",
            "data": json.dumps({
                "error": str(e),
            }),
        }


@app.post("/batch-generate", response_model=BatchGenerateResponse)
async def batch_generate(request: BatchGenerateRequest):
    """Process multiple prompts in a batch for higher throughput."""
    global inference_engine, total_tokens_generated
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Count input tokens
        input_tokens = sum(len(inference_engine.tokenizer.encode(prompt)) for prompt in request.prompts)
        
        # Batch processing
        generated_texts = inference_engine.batch_process(
            prompts=request.prompts,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Count output tokens
        output_tokens = sum(len(inference_engine.tokenizer.encode(text)) for text in generated_texts)
        
        # Update metrics
        total_tokens_generated += output_tokens
        
        return BatchGenerateResponse(
            generated_texts=generated_texts,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        logger.error(f"Error in batch generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/code", response_model=GenerateResponse)
async def complete_code(request: CodeCompletionRequest):
    """Complete code based on the prompt."""
    global inference_engine, total_tokens_generated
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Count input tokens
        input_tokens = len(inference_engine.tokenizer.encode(request.code_prompt))
        
        # Handle streaming response
        if request.stream:
            return EventSourceResponse(
                _complete_code_stream_events(
                    code_prompt=request.code_prompt,
                    language=request.language,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                ),
                media_type="text/event-stream",
            )
        
        # Standard code completion
        completed_code = inference_engine.complete_code(
            code_prompt=request.code_prompt,
            language=request.language,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Count output tokens
        output_tokens = len(inference_engine.tokenizer.encode(completed_code))
        
        # Update metrics
        total_tokens_generated += output_tokens
        
        return GenerateResponse(
            generated_text=completed_code,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        logger.error(f"Error completing code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _complete_code_stream_events(
    code_prompt: str,
    language: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    """Generate streaming events for code completion."""
    global inference_engine, total_tokens_generated
    
    try:
        # Initialize token counter
        token_index = 0
        
        # Generate tokens
        async for token in await asyncio.to_thread(
            inference_engine.complete_code,
            code_prompt=code_prompt,
            language=language,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=True,
        ):
            # Update metrics
            total_tokens_generated += 1
            
            # Yield token
            yield {
                "event": "token",
                "data": json.dumps({
                    "token": token,
                    "index": token_index,
                }),
            }
            
            # Increment token index
            token_index += 1
            
            # Small delay to avoid overwhelming the client
            await asyncio.sleep(0.01)
        
        # Signal end of stream
        yield {
            "event": "done",
            "data": json.dumps({
                "total_tokens": token_index,
            }),
        }
    except Exception as e:
        logger.error(f"Error in streaming code completion: {e}")
        yield {
            "event": "error",
            "data": json.dumps({
                "error": str(e),
            }),
        }


@app.post("/math", response_model=GenerateResponse)
async def solve_math(request: MathSolvingRequest):
    """Solve a mathematical problem."""
    global inference_engine, total_tokens_generated
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Count input tokens
        input_tokens = len(inference_engine.tokenizer.encode(request.problem))
        
        # Handle streaming response
        if request.stream:
            return EventSourceResponse(
                _solve_math_stream_events(
                    problem=request.problem,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    show_work=request.show_work,
                ),
                media_type="text/event-stream",
            )
        
        # Standard math solving
        solution = inference_engine.solve_math(
            problem=request.problem,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            show_work=request.show_work,
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Count output tokens
        output_tokens = len(inference_engine.tokenizer.encode(solution))
        
        # Update metrics
        total_tokens_generated += output_tokens
        
        return GenerateResponse(
            generated_text=solution,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        logger.error(f"Error solving math problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _solve_math_stream_events(
    problem: str,
    max_new_tokens: int,
    temperature: float,
    show_work: bool,
):
    """Generate streaming events for math problem solving."""
    global inference_engine, total_tokens_generated
    
    try:
        # Initialize token counter
        token_index = 0
        
        # Generate tokens
        async for token in await asyncio.to_thread(
            inference_engine.solve_math,
            problem=problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            show_work=show_work,
            stream=True,
        ):
            # Update metrics
            total_tokens_generated += 1
            
            # Yield token
            yield {
                "event": "token",
                "data": json.dumps({
                    "token": token,
                    "index": token_index,
                }),
            }
            
            # Increment token index
            token_index += 1
            
            # Small delay to avoid overwhelming the client
            await asyncio.sleep(0.01)
        
        # Signal end of stream
        yield {
            "event": "done",
            "data": json.dumps({
                "total_tokens": token_index,
            }),
        }
    except Exception as e:
        logger.error(f"Error in streaming math solution: {e}")
        yield {
            "event": "error",
            "data": json.dumps({
                "error": str(e),
            }),
        }


@app.post("/translate", response_model=GenerateResponse)
async def translate(request: TranslationRequest):
    """Translate text from source language to target language."""
    global inference_engine, total_tokens_generated
    
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Count input tokens
        input_tokens = len(inference_engine.tokenizer.encode(request.text))
        
        # Handle streaming response
        if request.stream:
            return EventSourceResponse(
                _translate_stream_events(
                    text=request.text,
                    source_lang=request.source_lang,
                    target_lang=request.target_lang,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                ),
                media_type="text/event-stream",
            )
        
        # Standard translation
        translation = inference_engine.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Count output tokens
        output_tokens = len(inference_engine.tokenizer.encode(translation))
        
        # Update metrics
        total_tokens_generated += output_tokens
        
        return GenerateResponse(
            generated_text=translation,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _translate_stream_events(
    text: str,
    source_lang: str,
    target_lang: str,
    max_new_tokens: int,
    temperature: float,
):
    """Generate streaming events for translation."""
    global inference_engine, total_tokens_generated
    
    try:
        # Initialize token counter
        token_index = 0
        
        # Generate tokens
        async for token in await asyncio.to_thread(
            inference_engine.translate,
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        ):
            # Update metrics
            total_tokens_generated += 1
            
            # Yield token
            yield {
                "event": "token",
                "data": json.dumps({
                    "token": token,
                    "index": token_index,
                }),
            }
            
            # Increment token index
            token_index += 1
            
            # Small delay to avoid overwhelming the client
            await asyncio.sleep(0.01)
        
        # Signal end of stream
        yield {
            "event": "done",
            "data": json.dumps({
                "total_tokens": token_index,
            }),
        }
    except Exception as e:
        logger.error(f"Error in streaming translation: {e}")
        yield {
            "event": "error",
            "data": json.dumps({
                "error": str(e),
            }),
        }


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for interactive text generation."""
    global inference_engine, total_tokens_generated
    
    await websocket.accept()
    
    try:
        if inference_engine is None:
            await websocket.send_json({"error": "Model not loaded"})
            await websocket.close()
            return
        
        # Receive request data
        data = await websocket.receive_json()
        
        # Extract parameters
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)
        repetition_penalty = data.get("repetition_penalty", 1.1)
        do_sample = data.get("do_sample", True)
        stop_strings = data.get("stop_strings")
        
        # Count input tokens
        input_tokens = len(inference_engine.tokenizer.encode(prompt))
        
        # Initialize token counter
        token_index = 0
        
        # Generate tokens
        for token in inference_engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_strings=stop_strings,
            stream=True,
        ):
            # Update metrics
            total_tokens_generated += 1
            
            # Send token
            await websocket.send_json({
                "token": token,
                "index": token_index,
            })
            
            # Increment token index
            token_index += 1
            
            # Check if client is still connected
            try:
                # Try to receive any message with a very short timeout
                await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
            except asyncio.TimeoutError:
                # No message received, continue
                pass
            except WebSocketDisconnect:
                # Client disconnected
                logger.info("WebSocket client disconnected")
                return
        
        # Signal end of stream
        await websocket.send_json({
            "done": True,
            "total_tokens": token_index,
        })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket generation: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Ensure WebSocket is closed
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    
    # Get API port from environment variable or use default
    port = int(os.environ.get("DEEPSHIVA_API_PORT", 8000))
    
    # Run API server
    uvicorn.run(
        "enhanced_app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
