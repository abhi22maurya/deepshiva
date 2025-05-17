<div align="center">
  <h1>DeepShiva</h1>
  <h3>Optimized Code Language Model Interface</h3>
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
  [![Documentation](https://img.shields.io/badge/Documentation-Read%20the%20Docs-blue)](docs/GETTING_STARTED.md)
</div>

## 🚀 Overview

DeepShiva is an optimized interface for running code language models, specifically designed for better performance on various hardware including Apple Silicon (M1/M2) with MPS acceleration. It provides a user-friendly web interface and API for interacting with models like CodeLlama and StableCode.

## ✨ Key Features

- **💻 Code Intelligence**: Advanced code generation, completion, and understanding
- **⚡ Efficient Architecture**: Optimized for various hardware including Apple Silicon
- **🖥️ User-friendly Interface**: Simple web UI for interacting with the model
- **💾 Memory Efficiency**: Options for 8-bit quantization to reduce memory usage
- **🔌 API Ready**: Easy integration via REST API
- **🚀 Minimal Dependencies**: Streamlined requirements for easy setup

## 🛠️ Supported Models

| Model | Parameters | Description |
|-------|------------|-------------|
| CodeLlama-7b | 7B | Meta's CodeLlama model for code generation |
| StableCode-3b | 3B | Stability AI's code generation model |
| Other HuggingFace models | Varies | Compatible with most causal language models |

## 🛠️ Tech Stack

- **Core**: PyTorch 2.0+
- **Inference**: Optimized Transformers
- **Frontend**: Flask
- **API**: Flask REST API
- **Deployment**: Python standalone

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB recommended for larger models)
- 4GB+ free disk space (more for model storage)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhi22maurya/deepshiva.git
   cd deepshiva
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv deepshiva_env
   source deepshiva_env/bin/activate  # On Windows: .\deepshiva_env\Scripts\activate
   ```

3. Install minimal dependencies:
   ```bash
   pip install -r requirements-minimal.txt
   ```

## 🏃‍♂️ Quick Examples

### Web Interface

```bash
python minimal_app.py
# Open http://localhost:5005 in your browser
```

### Python API

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

## 📚 Documentation

For detailed documentation, please refer to:

- [Getting Started](docs/GETTING_STARTED.md) - Setup and basic usage
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [Model Card](docs/MODEL_CARD.md) - Model details and compatibility
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or feedback, please open an issue on the GitHub repository.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CodeLlama](https://ai.meta.com/llama/)
- [StableCode](https://stability.ai/blog/stable-code-stable-lm-3b-code-completion-model)
- [PyTorch](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/)
