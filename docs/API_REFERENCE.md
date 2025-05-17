# API Reference

This document provides detailed information about the DeepShiva API endpoints and Python interface.

## Web API

The DeepShiva web API is implemented using Flask and is available at:
```
http://localhost:5005
```

## Flask API Endpoints

### Load Model

Load a model for inference.

**URL**: `/load_model`  
**Method**: `POST`

#### Request Body

```json
{
  "model_name": "stabilityai/stable-code-3b"
}
```

#### Parameters

| Parameter  | Type   | Required | Default | Description                                    |
|------------|--------|----------|---------|------------------------------------------------|
| model_name | string | No       | "stabilityai/stable-code-3b" | The model name or path to load |

#### Example Response

```json
{
  "status": "success",
  "message": "Model stabilityai/stable-code-3b loaded successfully"
}
```

### Generate Text

Generate text based on a prompt.

**URL**: `/generate`  
**Method**: `POST`

#### Request Body

```json
{
  "prompt": "def fibonacci(n):",
  "max_length": 512,
  "temperature": 0.7
}
```

#### Parameters

| Parameter    | Type   | Required | Default | Description                                    |
|--------------|--------|----------|---------|------------------------------------------------|
| prompt       | string | Yes      | -       | The input prompt                               |
| max_length   | int    | No       | 512     | Maximum number of tokens to generate          |
| temperature  | float  | No       | 0.7     | Controls randomness (0.0 to 1.0)              |

#### Example Response

```json
{
  "status": "success",
  "response": "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
}
```

### Stream Response

Generate text with streaming response.

**URL**: `/stream`  
**Method**: `POST`

#### Request Body

```json
{
  "prompt": "def fibonacci(n):",
  "max_length": 512,
  "temperature": 0.7
}
```

#### Parameters

Same as the `/generate` endpoint.

#### Response

Streams text as Server-Sent Events (SSE).

### Model Status

Check the status of the loaded model.

**URL**: `/model_status`  
**Method**: `GET`

#### Example Response

```json
{
  "status": "success",
  "model_loaded": true,
  "model_name": "stabilityai/stable-code-3b",
  "device": "cpu"
}
```

## Error Responses

### Error Response Format

```json
{
  "status": "error",
  "message": "Error message details"
}
```

### Common Error Scenarios

- **Model Not Loaded**: Attempt to generate text before loading a model
- **Invalid Parameters**: Providing invalid parameters (e.g., negative max_length)
- **Out of Memory**: System runs out of memory while loading or running the model
- **Model Not Found**: Specified model cannot be found or downloaded

## Python API

The `OptimizedLLM` class provides a Python interface for using the language models.

### Initialization

```python
from optimized_llm import OptimizedLLM

llm = OptimizedLLM(
    model_name="codellama/CodeLlama-7b-hf",  # Model name or path
    device="auto",                          # "auto", "cuda", "mps", or "cpu"
    max_memory=None,                       # Memory configuration for model sharding
    load_in_8bit=False,                    # Whether to use 8-bit quantization
    torch_dtype=None                       # Override default torch dtype
)
```

### Methods

#### Generate Text

```python
response = llm.generate(
    prompt="def fibonacci(n):",        # Input prompt
    max_length=512,                  # Maximum tokens to generate
    temperature=0.7,                 # Sampling temperature
    top_p=0.9,                       # Nucleus sampling parameter
    top_k=50,                        # Top-k sampling parameter
    repetition_penalty=1.1,          # Penalty for repeating tokens
    stop_sequences=None,             # List of sequences to stop on
    stream=False                     # Whether to stream the output
)
```

#### Clear Memory

```python
llm.clear_memory()
```

Releases memory used by the model.
