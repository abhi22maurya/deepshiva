# DeepShiva Model Card

## Interface Details

- **Interface Name**: DeepShiva Optimized LLM Interface
- **Version**: 1.0.0
- **Release Date**: May 2024
- **License**: Apache 2.0
- **Repository**: [GitHub](https://github.com/abhi22maurya/deepshiva)

## Interface Overview

DeepShiva provides an optimized interface for running code language models with better performance on various hardware platforms including Apple Silicon. It's designed to make code generation models more accessible with minimal setup and resource requirements.

## Supported Models

DeepShiva is compatible with most causal language models from Hugging Face. Recommended models include:

### Primary Supported Models

- **CodeLlama-7b** - Meta's CodeLlama model for code generation
- **StableCode-3b** - Stability AI's code generation model

### Other Compatible Models

- Most decoder-only (causal) language models from Hugging Face
- Models that are compatible with the Transformers library

## Intended Use

### Primary Use Cases

- Code completion and generation
- Code explanation and documentation
- Programming assistance
- Educational tools for programming

### Out-of-Scope Uses

- Medical or legal advice
- Generating harmful or misleading content
- High-stakes decision making without human oversight

## Interface Architecture

- **Core Framework**: PyTorch with Transformers
- **Web Interface**: Flask-based web UI
- **API**: REST API for integration
- **Optimization**: Memory-efficient loading and inference
- **Hardware Support**: 
  - NVIDIA GPUs (CUDA)
  - Apple Silicon (MPS)
  - CPU fallback for all platforms

## Features

### Optimizations

- **Memory Efficiency**: Options for 8-bit quantization
- **Device Detection**: Automatic selection of best available device
- **Error Handling**: Robust error handling and recovery
- **Streaming**: Support for streaming generation responses

### User Interface

- **Web UI**: Simple interface for model interaction
- **API**: Programmatic access via REST API
- **Python Interface**: Direct integration in Python code

## Model Performance

Performance depends on the specific model used. Here are general characteristics of the recommended models:

### CodeLlama-7b

- **HumanEval**: ~30-35% pass@1
- **Good at**: Python, JavaScript, Java, C++
- **Context Length**: 16K tokens
- **Memory Usage**: ~14GB in full precision, ~7GB with 8-bit quantization

### StableCode-3b

- **HumanEval**: ~25-30% pass@1
- **Good at**: Python, JavaScript
- **Context Length**: 4K tokens
- **Memory Usage**: ~6GB in full precision, ~3GB with 8-bit quantization

### Limitations

- Models may generate incorrect or nonsensical code
- Limited understanding of very recent libraries/frameworks
- Performance varies across programming languages
- May exhibit biases present in training data

## Ethical Considerations

### Potential Biases

- Models may inherit biases from their training data
- Code generation may reflect biases in existing codebases
- Performance may vary across different programming languages

### Recommendations

- Always review generated code before use
- Be cautious when using for critical applications
- Consider the context and potential biases in outputs
- Use the models as assistants rather than replacements for human judgment

## Deployment

### Hardware Requirements

#### Minimum
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 4GB+ free space

#### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM or Apple Silicon M1/M2
- **RAM**: 16GB+
- **Storage**: 20GB+ SSD

### Usage Example

```python
from optimized_llm import OptimizedLLM

# Initialize the model
llm = OptimizedLLM(
    model_name="codellama/CodeLlama-7b-hf",
    device="auto",  # Will use MPS on Mac, CUDA on NVIDIA, or CPU
    load_in_8bit=True  # Enable 8-bit quantization for reduced memory usage
)

# Generate code
response = llm.generate(
    prompt="def fibonacci(n):",
    max_length=100,
    temperature=0.2
)

print(response)
```

## Maintenance

### Support

- GitHub Issues: [https://github.com/abhi22maurya/deepshiva/issues](https://github.com/abhi22maurya/deepshiva/issues)
- Documentation: See the `docs/` directory in the repository

### Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**
   - Try reducing `max_length`
   - Close other memory-intensive applications
   - Enable 8-bit quantization if not already enabled
   - Use a smaller model (e.g., switch from 7B to 3B)

2. **Slow Performance**
   - Ensure you're using the appropriate device (check console output)
   - Reduce `max_length` and `top_k` parameters
   - Use a smaller model if available

## License

Apache License 2.0

## Contact

For questions or feedback, please open an issue on the GitHub repository.
