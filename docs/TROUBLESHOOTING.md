# Troubleshooting Guide

This guide addresses common issues you might encounter when using DeepShiva and provides solutions to resolve them.

## Model Loading Issues

### Problem: Model fails to load

**Symptoms:**
- Error message about model not found
- Application crashes during model loading
- Timeout during model initialization

**Solutions:**
1. **Check internet connection** - Models are downloaded from Hugging Face on first use
2. **Verify model name** - Ensure the model name is correct (e.g., `codellama/CodeLlama-7b-hf`)
3. **Check disk space** - Models can be several GB in size
4. **Try a smaller model** - If memory is limited, try `stabilityai/stable-code-3b` instead

### Problem: Out of memory during model loading

**Symptoms:**
- "CUDA out of memory" error
- "Not enough memory" error
- System becomes unresponsive

**Solutions:**
1. **Enable 8-bit quantization** - Set `load_in_8bit=True` when initializing the model
2. **Close other applications** - Free up system memory
3. **Use a smaller model** - Try a 3B parameter model instead of 7B
4. **Use CPU mode** - Set `device="cpu"` for reliable (but slower) operation

## Generation Issues

### Problem: Generation is too slow

**Symptoms:**
- Long waiting times for responses
- System becomes unresponsive during generation

**Solutions:**
1. **Reduce max_length** - Lower the maximum generation length
2. **Use a smaller model** - Smaller models are faster
3. **Optimize parameters** - Lower values for `top_k` and `top_p`
4. **Check device** - Ensure you're using GPU/MPS acceleration if available

### Problem: Poor quality generations

**Symptoms:**
- Nonsensical or irrelevant code
- Incomplete responses
- Repetitive text

**Solutions:**
1. **Adjust temperature** - Lower temperature (0.1-0.3) for more focused outputs
2. **Improve prompts** - Be more specific in your prompts
3. **Increase repetition penalty** - Try values between 1.1 and 1.5
4. **Try a different model** - Some models perform better for certain tasks

## Web Interface Issues

### Problem: Web interface not loading

**Symptoms:**
- Browser shows "This site can't be reached"
- Connection refused errors

**Solutions:**
1. **Check if server is running** - Look for console output showing the server started
2. **Verify port** - Make sure port 5005 is not being used by another application
3. **Try a different browser** - Some browsers may have compatibility issues
4. **Check firewall settings** - Ensure the application is allowed through your firewall

### Problem: Interface loads but model doesn't respond

**Symptoms:**
- "Model not loaded" error
- Timeout when generating responses

**Solutions:**
1. **Load the model first** - Click "Load Model" before attempting generation
2. **Check console output** - Look for error messages in the terminal
3. **Restart the application** - Sometimes a fresh start resolves issues
4. **Verify model compatibility** - Not all models work with the interface

## Python API Issues

### Problem: Import errors

**Symptoms:**
- `ModuleNotFoundError` or `ImportError`
- Missing dependencies

**Solutions:**
1. **Install requirements** - Run `pip install -r requirements-minimal.txt`
2. **Check Python version** - Ensure you're using Python 3.8+
3. **Verify virtual environment** - Make sure your virtual environment is activated

### Problem: API usage errors

**Symptoms:**
- TypeError or ValueError when calling functions
- Unexpected behavior

**Solutions:**
1. **Check documentation** - Verify the correct function signatures
2. **Update code** - Ensure you're using the latest version of the API
3. **Debug with print statements** - Track the flow of execution

## Hardware-Specific Issues

### NVIDIA GPU Issues

**Symptoms:**
- CUDA errors
- GPU not detected

**Solutions:**
1. **Update drivers** - Ensure you have the latest NVIDIA drivers
2. **Check CUDA installation** - Verify CUDA toolkit is properly installed
3. **Monitor GPU usage** - Use `nvidia-smi` to check GPU memory and utilization

### Apple Silicon (M1/M2) Issues

**Symptoms:**
- MPS device not found
- Slow performance on Mac

**Solutions:**
1. **Update PyTorch** - Ensure you have PyTorch 2.0+ with MPS support
2. **Check macOS version** - MPS requires macOS 12.3+
3. **Use CPU fallback** - If MPS is unstable, try `device="cpu"`

## Getting Additional Help

If you're still experiencing issues:

1. **Check the console output** - Error messages often provide useful information
2. **Search GitHub issues** - Your problem may have been solved already
3. **Open a new issue** - Provide detailed information about your problem
4. **Include system information** - OS, hardware, Python version, and error messages

Remember that large language models are resource-intensive. Performance will vary based on your hardware capabilities and the specific models you're using.
