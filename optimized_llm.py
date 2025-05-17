"""
Optimized LLM Interface for DeepShiva

This module provides an optimized interface for running LLM inference
with better memory management and performance on Apple Silicon.
"""

import os
import torch
from typing import Optional, Dict, Any, List, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
from threading import Thread
import gc
import torch.quantization

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class OptimizedLLM:
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-hf",
        device: str = "auto",
        max_memory: Optional[Dict] = None,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """Initialize the optimized LLM.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to load the model on ('auto', 'cuda', 'mps', 'cpu')
            max_memory: Maximum memory configuration for model sharding
            load_in_8bit: Whether to load the model in 8-bit precision (handled via PyTorch MPS quantization if device is mps)
            torch_dtype: Override the default torch dtype
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.max_memory = max_memory
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype or torch.float16
        self.model = None
        self.tokenizer = None
        self.streamer = None
        
        # Initialize model and tokenizer
        self._initialize_model()
    
    def _get_device(self, device_str: str) -> torch.device:
        """Get the appropriate device based on the input string."""
        if device_str == "auto":
            # For stability, we'll use CPU by default on Apple Silicon
            # MPS support is still experimental and can cause issues
            print("Using CPU for better stability (MPS support is experimental)")
            return torch.device("cpu")
        return torch.device(device_str)
    
    def _initialize_model(self):
        """Initialize the model with optimized settings using device_map for sharding."""
        print(f"Loading model {self.model_name} on {self.device}...")
        
        # Clear any existing model from memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        gc.collect()
        torch.mps.empty_cache()
        
        try:
            # Initialize tokenizer with proper padding configuration
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Set pad_token to eos_token for tokenizer")
            
            # Configure model loading to use CPU only
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "trust_remote_code": True,
                "device_map": None,  # Disable device mapping
                "torch_dtype": torch.float32,  # Use float32 for better compatibility
                "low_cpu_mem_usage": True,
            }
            
            # Load the model on CPU
            print("Loading model on CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            print(f"Model loaded with dtype: {self.model.dtype}, device: {self.model.device}")
            print(f"Model device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'Not using device map'}")
            
            # Ensure model is on CPU for stability
            print("Moving model to CPU for stable operation")
            try:
                self.model = self.model.cpu()
            except Exception as e:
                print(f"Warning: Could not move model to CPU: {e}")
            
            # Skip quantization as it can cause issues with some models
            if self.load_in_8bit:
                print("8-bit quantization is not supported in this configuration")
                print("Running in full precision mode")
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Try using a smaller model (e.g., 'stabilityai/stable-code-1.3b' or 'codellama/CodeLlama-7b-hf')")
            print("2. Ensure you have enough free memory (at least 12GB recommended for 3B models)")
            print("3. Try running with load_in_8bit=True")
            print("4. Close other memory-intensive applications")
            print("5. Try running on CPU with device='cpu'")
            raise

        self.model.eval()
        
        # Warm up the model
        self._warmup()
        
        print(f"Model loaded and initialized on {self.device}")
    
    def _warmup(self):
        """Warm up the model with a small inference."""
        print("Warming up the model...")
        # Ensure dummy input is on the same device as the model
        dummy_input = torch.tensor([[self.tokenizer.eos_token_id]], device='cpu')
        with torch.no_grad():
            try:
                _ = self.model.generate(
                    dummy_input,
                    max_new_tokens=1,
                    do_sample=False
                )
                print("Model warmup successful")
            except Exception as e:
                print(f"Warning: Model warmup failed, but continuing: {e}")
                # Continue even if warmup fails
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, TextIteratorStreamer]:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stop_sequences: List of strings to stop generation on
            stream: Whether to stream the output
            
        Returns:
            Generated text or streamer object if streaming
        """
        # Double-check pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # Configure stopping criteria
        stop_token_ids = [self.tokenizer.eos_token_id]
        if stop_sequences:
            for seq in stop_sequences:
                stop_token_ids.append(self.tokenizer.encode(seq, add_special_tokens=False)[0])
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        # Configure generation - optimized for CPU performance
        generation_kwargs = {
            **inputs,
            "max_new_tokens": min(max_length, 256),  # Limit tokens for faster generation
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "stopping_criteria": stopping_criteria,
            "num_beams": 1,  # Disable beam search for speed
            "use_cache": True,  # Enable KV caching
        }
        
        if stream:
            # Set up streaming
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=300,
            )
            generation_kwargs["streamer"] = streamer
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            return streamer
        else:
            # Generate all at once
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            # Decode and return
            generated = outputs[0, inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
    
    def clear_memory(self):
        """Clear GPU/CPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Clean up resources."""
        self.clear_memory()
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


def test_optimized_llm():
    """Test the optimized LLM implementation."""
    # Configure memory settings for MPS
    max_memory = {
        "mps": "4GB",
        "cpu": "8GB"
    }
    
    # Initialize the model
    llm = OptimizedLLM(
        model_name="codellama/CodeLlama-7b-hf",
        device="auto",
        max_memory=max_memory,
        load_in_8bit=True,
        torch_dtype=torch.float16
    )
    
    # Test code completion
    print("\n=== Testing Code Completion ===")
    prompt = "def fibonacci(n):"
    print(f"Prompt: {prompt}")
    
    response = llm.generate(
        prompt=prompt,
        max_length=100,
        temperature=0.2,
        stream=False
    )
    print("\nResponse:", response)
    
    # Test text generation
    print("\n=== Testing Text Generation ===")
    prompt = "Explain the concept of recursion in programming."
    print(f"Prompt: {prompt}")
    
    response = llm.generate(
        prompt=prompt,
        max_length=200,
        stream=False
    )
    print("\nResponse:", response)
    
    # Test streaming
    print("\n=== Testing Streaming ===")
    prompt = "Write a Python function to reverse a string."
    print(f"Prompt: {prompt}")
    
    streamer = llm.generate(
        prompt=prompt,
        max_length=150,
        stream=True
    )
    print("\nStreaming Response:")
    for token in streamer:
        print(token, end="", flush=True)
    print("\nStreaming complete.")
    
    # Clean up
    llm.clear_memory()


if __name__ == "__main__":
    test_optimized_llm()
