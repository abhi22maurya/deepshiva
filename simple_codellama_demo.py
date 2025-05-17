"""
Simple demo for CodeLlama model.

This script demonstrates how to load and use the CodeLlama model
for text generation and code completion.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_codellama():
    """Load the CodeLlama model and tokenizer."""
    model_path = "models/pretrained/codellama-7b"
    print(f"Loading CodeLlama model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with device map for automatic device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=200, temperature=0.7):
    """Generate text using the model."""
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def test_code_completion(model, tokenizer):
    """Test code completion with the model."""
    print("\n=== Testing Code Completion ===")
    
    # Code completion prompt
    prompt = """def fibonacci(n):
    """
    
    print(f"Prompt: {prompt}")
    
    # Generate completion
    print("Generating completion...")
    completion = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_length=100,
        temperature=0.2,
    )
    
    print(f"\nCompletion:\n{completion}")

def test_text_generation(model, tokenizer):
    """Test text generation with the model."""
    print("\n=== Testing Text Generation ===")
    
    # Text generation prompt
    prompt = "Explain how to implement a binary search tree in Python."
    
    print(f"Prompt: {prompt}")
    
    # Generate text
    print("Generating response...")
    response = generate_text(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_length=300,
        temperature=0.7,
    )
    
    print(f"\nResponse:\n{response}")

def main():
    """Main function to run the demo."""
    # Load model and tokenizer
    model, tokenizer = load_codellama()
    
    # Run tests
    test_code_completion(model, tokenizer)
    test_text_generation(model, tokenizer)
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
