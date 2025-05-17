"""
Optimized CodeLlama Demo for Mac

This script demonstrates how to efficiently run CodeLlama on Mac hardware
using memory optimization techniques.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import psutil
import platform
import sys

def print_system_info():
    """Print system information for debugging."""
    print("\n=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        print("Using CPU")

def load_codellama_optimized():
    """Load the CodeLlama model with optimizations for Mac."""
    model_path = "models/pretrained/codellama-7b"
    print(f"\nLoading CodeLlama model from {model_path}...")
    
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Configure device map
    device_map = "auto"
    
    try:
        # Try to use MPS (Metal Performance Shaders) on Apple Silicon
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.mps.set_per_process_memory_fraction(0.9)  # Limit memory usage
            print("Using MPS (Metal) for acceleration")
        # Fall back to CUDA if available
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
            print("Using CUDA for acceleration")
        else:
            device = torch.device("cpu")
            print("Using CPU (no GPU acceleration available)")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_folder="offload"  # Folder for offloading layers if needed
        )
        
        # Move model to device if not using device_map
        if not hasattr(model, "hf_device_map"):
            model = model.to(device)
        
        print("Model loaded successfully with optimizations!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nTrying with reduced precision...")
        
        # Fall back to 8-bit quantization if 4-bit fails
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=True,
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder="offload"
            )
            print("Model loaded with 8-bit quantization")
            return model, tokenizer
        except Exception as e2:
            print(f"Failed to load with 8-bit: {str(e2)}")
            raise

def generate_text_optimized(prompt, model, tokenizer, max_length=200, temperature=0.7):
    """Generate text with optimized settings."""
    try:
        # Encode the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate with optimized settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                early_stopping=True
            )
        
        # Decode and return the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Out of memory! Try reducing max_length or batch size.")
        raise

def test_code_completion_optimized(model, tokenizer):
    """Test code completion with optimizations."""
    print("\n=== Testing Code Completion ===")
    
    prompt = """def fibonacci(n):
    """
    
    print(f"Prompt: {prompt}")
    print("Generating completion (this may take a moment)...")
    
    try:
        completion = generate_text_optimized(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_length=100,
            temperature=0.2,
        )
        print(f"\nCompletion:\n{completion}")
    except Exception as e:
        print(f"Error during generation: {str(e)}")

def test_text_generation_optimized(model, tokenizer):
    """Test text generation with optimizations."""
    print("\n=== Testing Text Generation ===")
    
    prompt = "Explain how to implement a binary search tree in Python."
    
    print(f"Prompt: {prompt}")
    print("Generating response (this may take a moment)...")
    
    try:
        response = generate_text_optimized(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            max_length=200,
            temperature=0.7,
        )
        print(f"\nResponse:\n{response}")
    except Exception as e:
        print(f"Error during generation: {str(e)}")

def main():
    """Main function to run the optimized demo."""
    # Print system information
    print_system_info()
    
    try:
        # Load model with optimizations
        model, tokenizer = load_codellama_optimized()
        
        # Run tests
        test_code_completion_optimized(model, tokenizer)
        test_text_generation_optimized(model, tokenizer)
        
        print("\n=== Demo Complete ===")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have enough free memory (at least 16GB RAM recommended)")
        print("2. Try closing other memory-intensive applications")
        print("3. Consider using a smaller model if available")
        print("4. Check if you're using the latest version of PyTorch and transformers")

if __name__ == "__main__":
    main()
