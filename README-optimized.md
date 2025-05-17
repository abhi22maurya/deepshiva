# DeepShiva - Optimized CodeLlama Interface

This is an optimized version of the DeepShiva project, specifically designed for better performance on Mac with MPS (Metal Performance Shaders) support.

## Features

- 🚀 Optimized for Apple Silicon (M1/M2) with MPS acceleration
- 💾 Memory-efficient model loading with 8-bit quantization
- 🖥️ User-friendly Gradio web interface
- ⚡ Streamlined codebase focused on inference performance
- 🛠️ Easy-to-use API for integration with other applications

## Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements-minimal.txt
   ```

2. **Run the Web Interface**

   ```bash
   python app.py
   ```

   This will start a local web server. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:7860).

## Usage

### Web Interface

The web interface provides an easy way to interact with the model:

1. Enter the model name/path (default is `codellama/CodeLlama-7b-hf`)
2. Click "Load Model" to load the model (this may take a few minutes)
3. Enter your prompt in the text area
4. Adjust generation parameters as needed
5. Click "Generate" or press Enter to get a response

### Python API

You can also use the optimized model directly in your Python code:

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
2. **8-bit Quantization**: Enabled by default to reduce memory usage with minimal impact on output quality.
3. **Context Length**: Keep prompts reasonably sized for best performance.
4. **Temperature**: Lower values (0.1-0.3) produce more focused outputs, while higher values (0.7-1.0) increase creativity.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Try reducing `max_length`
   - Close other memory-intensive applications
   - Enable 8-bit quantization if not already enabled

2. **Slow Performance**
   - Ensure you're using MPS (check console output for device being used)
   - Reduce `max_length` and `top_k` parameters
   - Use a smaller model if available

3. **Model Loading Failures**
   - Check your internet connection if downloading the model
   - Verify you have enough disk space (models can be several GB)
   - Ensure you have the latest version of dependencies

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CodeLlama](https://ai.meta.com/llama/)
- [Gradio](https://gradio.app/)

---

For questions or support, please open an issue on the GitHub repository.
