"""
Run CodeLlama with the enhanced inference engine.

This script demonstrates how to use the enhanced inference engine with the CodeLlama model.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inference.enhanced_inference import EnhancedInference

# Configuration
MODEL_PATH = "models/pretrained/codellama-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = "fp16" if DEVICE == "cuda" else "fp32"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2  # Lower temperature for more focused outputs

def load_codellama():
    """Load the CodeLlama model and tokenizer."""
    print(f"Loading CodeLlama model from {MODEL_PATH}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Configure device map
    device_map = "auto"
    
    # Load model with appropriate precision
    if PRECISION == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
    
    print(f"Model loaded on {device_map} with {PRECISION} precision")
    return model, tokenizer

class CodeLlamaWrapper(torch.nn.Module):
    """Wrapper class to make CodeLlama compatible with our enhanced inference."""
    
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.device = next(model.parameters()).device
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True, **kwargs):
        """Forward pass that matches the expected interface."""
        # Prepare inputs
        inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        
        # Add past_key_values if provided
        if past_key_values is not None:
            inputs["past_key_values"] = past_key_values
        
        # Forward pass
        outputs = self.model(**inputs, use_cache=use_cache, output_hidden_states=False)
        
        # Return in the expected format
        return {
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values if use_cache else None
        }
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def generate(self, input_ids, attention_mask=None, max_length=100, **kwargs):
        """Generate method that matches the expected interface."""
        # Prepare inputs
        inputs = {
            "input_ids": input_ids,
            "max_length": max_length,
            "do_sample": True,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask
        
        # Generate
        outputs = self.model.generate(**inputs)
        return outputs

def test_code_completion(inference):
    """Test code completion with CodeLlama."""
    print("\n=== Testing Code Completion ===")
    
    # Code completion prompt
    prompt = """def fibonacci(n):
    """
    
    print(f"Prompt: {prompt}")
    
    # Generate completion
    print("Generating completion...")
    completion = inference.complete_code(
        code_prompt=prompt,
        language="python",
        max_new_tokens=100,
        temperature=0.2,
    )
    
    print(f"\nCompletion:\n{completion}")

def test_text_generation(inference):
    """Test text generation with CodeLlama."""
    print("\n=== Testing Text Generation ===")
    
    # Text generation prompt
    prompt = "Explain how to implement a binary search tree in Python."
    
    print(f"Prompt: {prompt}")
    
    # Generate text
    print("Generating response...")
    response = inference.generate(
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.7,
        stream=False,
    )
    
    print(f"\nResponse:\n{response}")

def main():
    """Main function to run the demo."""
    # Load model and tokenizer
    model, tokenizer = load_codellama()
    
    # Create wrapper
    wrapper = CodeLlamaWrapper(model, tokenizer)
    
    # Create enhanced inference instance
    inference = EnhancedInference(
        model_path=MODEL_PATH,
        device=DEVICE,
        precision=PRECISION,
    )
    
    # Replace the model and tokenizer with our loaded ones
    inference.model = wrapper
    inference.tokenizer = tokenizer
    
    # Run tests
    test_code_completion(inference)
    test_text_generation(inference)
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
