"""
Optimized CodeLlama Demo for Mac with MPS (Metal Performance Shaders)

This version is heavily optimized for Mac with Apple Silicon (M1/M2) and uses
MPS for acceleration with 4-bit quantization for better performance.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import platform
import psutil
import sys
import time
from tqdm import tqdm
from typing import Optional, Tuple, List

# Configuration
MAX_TOKENS = 100  # Reduced from 200 for faster response
TEMPERATURE = 0.7  # Slightly higher for more creative but focused output
TOP_P = 0.9  # Nucleus sampling
TOP_K = 50  # Top-k sampling
REPETITION_PENALTY = 1.1  # Slightly reduce repetition

def print_system_info():
    """Print system information for debugging."""
    print("\n=== System Information ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        print("MPS not available. Using CPU.")

def load_model_safely(model_path: str, device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model with 4-bit quantization for better performance."""
    print(f"\nLoading model from {model_path}...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # First load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"  # For batch processing
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    
    # Configure model loading
    print("Loading model with 4-bit quantization (this may take a few minutes)...")
    
    try:
        # Try loading with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            attn_implementation="eager"  # More stable with MPS
        )
        print("Model loaded with 4-bit quantization!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading with 4-bit quantization: {str(e)}")
        print("Falling back to standard loading...")
        
        # Fall back to standard loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        print("Model loaded with standard settings.")
        return model, tokenizer

def generate_text_streaming(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_TOKENS,
    temperature: float = TEMPERATURE
) -> str:
    """Generate text with streaming output and optimized performance."""
    print("\nGenerating response...")
    
    # Encode the prompt with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
    
    # Generate with optimized parameters
    start_time = time.time()
    generated = []
    
    try:
        with torch.no_grad():
            # Generate in a single pass for better performance
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            # Get the generated tokens
            generated_tokens = outputs.sequences[0, inputs.input_ids.shape[1]:]
            
            # Stream the output
            for token in tqdm(generated_tokens, desc="Generating"):
                token_item = token.item()
                generated.append(token_item)
                print(tokenizer.decode([token_item], skip_special_tokens=True), end="", flush=True)
                
                # Stop if we hit the end token
                if token_item == tokenizer.eos_token_id:
                    break
                    
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
    finally:
        elapsed = time.time() - start_time
        tokens_per_sec = len(generated) / elapsed if elapsed > 0 else 0
        print(f"\n\nGenerated {len(generated)} tokens in {elapsed:.2f}s ({tokens_per_sec:.2f} tokens/s)")
    
    # Return the full generated text
    return tokenizer.decode(generated, skip_special_tokens=True)

def test_code_completion(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """Test code completion with the model."""
    print("\n=== Testing Code Completion ===")
    
    prompt = """def fibonacci(n):
    """
    
    print(f"Prompt: {prompt}")
    
    # Generate completion with lower temperature for more deterministic output
    start_time = time.time()
    completion = generate_text_streaming(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_TOKENS,
        temperature=0.2,  # Lower temperature for code completion
    )
    
    elapsed = time.time() - start_time
    print(f"\n=== Generation Complete in {elapsed:.2f}s ===")

def test_text_generation(model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    """Test text generation with the model."""
    print("\n=== Testing Text Generation ===")
    
    prompt = "Explain how to implement a binary search tree in Python."
    
    print(f"Prompt: {prompt}")
    
    # Generate text with higher temperature for more creative output
    start_time = time.time()
    response = generate_text_streaming(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_TOKENS,
        temperature=0.7,  # Slightly higher for more creative text
    )
    
    elapsed = time.time() - start_time
    print(f"\n=== Generation Complete in {elapsed:.2f}s ===")

def main():
    """Main function to run the demo with optimizations."""
    # Print system information
    print_system_info()
    
    # Set device to MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Enable memory efficient attention if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.backends.mps.allocator_settings = {
                'max_split_size_mb': 128,
                'garbage_collection_threshold': 0.8
            }
            print("Enabled MPS memory optimizations")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU.")
    
    model_path = "models/pretrained/codellama-7b"
    
    try:
        # Load model with optimizations
        model, tokenizer = load_model_safely(model_path, device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Warm up the model
        print("\nWarming up the model...")
        with torch.no_grad():
            _ = model.generate(
                torch.tensor([[1]], device=model.device),
                max_new_tokens=1,
                do_sample=False
            )
        
        # Run tests
        test_code_completion(model, tokenizer)
        test_text_generation(model, tokenizer)
        
        print("\n=== Demo Complete ===")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
