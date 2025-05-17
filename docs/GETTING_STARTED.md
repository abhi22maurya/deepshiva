# Getting Started with DeepShiva

This guide will help you set up and start using DeepShiva, an optimized interface for running code language models with better performance on various hardware platforms including Apple Silicon.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- For GPU acceleration:
  - NVIDIA GPU: CUDA 11.7+ and compatible drivers
  - Apple Silicon: macOS 12.3+ with PyTorch 2.0+ (MPS support)
- At least 8GB RAM (16GB recommended for larger models)
- At least 4GB free disk space (more for model storage)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhi22maurya/deepshiva.git
   cd deepshiva
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv deepshiva_env
   source deepshiva_env/bin/activate  # On Windows: .\deepshiva_env\Scripts\activate
   ```

3. Install minimal dependencies:
   ```bash
   pip install -r requirements-minimal.txt
   ```

## Supported Models

DeepShiva is compatible with most causal language models from Hugging Face. The models are automatically downloaded when first used. Some recommended models include:

- `codellama/CodeLlama-7b-hf` - Meta's CodeLlama model for code generation
- `stabilityai/stable-code-3b` - Stability AI's code generation model

The first time you use a model, it will be downloaded from Hugging Face and cached locally.

## Running the Application

### Web Interface

Start the Flask web interface:

```bash
python minimal_app.py
```

Then open http://localhost:5005 in your browser.

### Using the Web Interface

1. Enter the model name/path (default is `stabilityai/stable-code-3b`)
2. Click "Load Model" to load the model (this may take a few minutes)
3. Enter your prompt in the text area
4. Click "Generate" to get a response

## Using the Python API

You can use the optimized model directly in your Python code:

```python
from optimized_llm import OptimizedLLM

# Initialize the model
llm = OptimizedLLM(
    model_name="codellama/CodeLlama-7b-hf",
    device="auto",  # Will use MPS on Mac, CUDA on NVIDIA, or CPU
    load_in_8bit=True  # Enable 8-bit quantization for reduced memory usage
)

# Generate text
response = llm.generate(
    prompt="def fibonacci(n):",
    max_length=100,
    temperature=0.2,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1
)

print(response)
```

## Performance Tips

1. **Model Size**: The 7B parameter model is recommended for most Macs. Larger models may require more memory.
2. **8-bit Quantization**: Enable `load_in_8bit=True` to reduce memory usage with minimal impact on output quality.
3. **Context Length**: Keep prompts reasonably sized for best performance.
4. **Temperature**: Lower values (0.1-0.3) produce more focused outputs, while higher values (0.7-1.0) increase creativity.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Try reducing `max_length`
   - Close other memory-intensive applications
   - Enable 8-bit quantization if not already enabled
   - Use a smaller model (e.g., switch from 7B to 3B)

2. **Slow Performance**
   - Ensure you're using the appropriate device (check console output)
   - Reduce `max_length` and `top_k` parameters
   - Use a smaller model if available

3. **Model Loading Failures**
   - Check your internet connection if downloading the model
   - Verify you have enough disk space (models can be several GB)
   - Ensure you have the latest version of dependencies

## Getting Help

If you encounter any issues, please:
1. Check the console output for error messages
2. Search the [GitHub issues](https://github.com/abhi22maurya/deepshiva/issues)
3. Open a new issue if your problem isn't addressed
