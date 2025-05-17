"""
Enhanced DeepShiva Inference Module

This module provides optimized inference capabilities for the DeepShiva model:
- Batched processing for higher throughput
- Streaming generation for faster user experience
- Speculative decoding for performance
- Optimized memory usage
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Union, Any, Iterator, Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Import model implementations
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.enhanced_model import EnhancedDeepShivaMoE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EnhancedInference:
    """Enhanced inference class for DeepShiva model with optimized performance."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "fp16",
        max_context_length: int = 8192,
        use_kv_cache: bool = True,
        batch_size: int = 1,
        use_bettertransformer: bool = True,
        use_compile: bool = False,
    ):
        """Initialize the enhanced inference engine.
        
        Args:
            model_path: Path to the model checkpoint
            device: Device to run inference on ("cuda" or "cpu")
            precision: Precision to use for inference ("fp32", "fp16", "int8", "int4")
            max_context_length: Maximum context length for the model
            use_kv_cache: Whether to use KV cache for faster generation
            batch_size: Default batch size for processing
            use_bettertransformer: Whether to use BetterTransformer for optimized inference
            use_compile: Whether to use torch.compile for optimized inference
        """
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.max_context_length = max_context_length
        self.use_kv_cache = use_kv_cache
        self.batch_size = batch_size
        self.use_bettertransformer = use_bettertransformer
        self.use_compile = use_compile
        
        logger.info(f"Loading model from {model_path}")
        self._load_model()
        logger.info("Model loaded successfully")
    
    def _load_model(self):
        """Load the model and tokenizer with optimizations."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate precision
        if self.precision == "fp32":
            self.model = EnhancedDeepShivaMoE.from_pretrained(self.model_path)
        elif self.precision == "fp16":
            self.model = EnhancedDeepShivaMoE.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None,
            )
        elif self.precision == "int8":
            self.model = EnhancedDeepShivaMoE.from_pretrained(
                self.model_path,
                load_in_8bit=True,
                device_map="auto",
            )
        elif self.precision == "int4":
            self.model = EnhancedDeepShivaMoE.from_pretrained(
                self.model_path,
                load_in_4bit=True,
                device_map="auto",
            )
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")
        
        # Move model to device if not using device_map
        if "device_map" not in self.precision and self.device != "auto":
            self.model.to(self.device)
        
        # Apply BetterTransformer if requested
        if self.use_bettertransformer:
            try:
                from optimum.bettertransformer import BetterTransformer
                self.model = BetterTransformer.transform(self.model)
                logger.info("Applied BetterTransformer optimization")
            except ImportError:
                logger.warning("BetterTransformer not available, skipping optimization")
        
        # Apply torch.compile if requested
        if self.use_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stop_strings: Optional[List[str]] = None,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Union[str, List[str], Iterator[str]]:
        """Generate text based on the prompt with streaming support.
        
        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            num_return_sequences: Number of sequences to return
            stop_strings: List of strings to stop generation when encountered
            stream: Whether to stream the output token by token
            callback: Optional callback function for streaming mode
            
        Returns:
            Generated text, list of generated texts, or iterator of tokens
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Truncate if needed
        if inputs.input_ids.shape[1] > self.max_context_length:
            logger.warning(f"Prompt is too long, truncating to {self.max_context_length} tokens")
            inputs.input_ids = inputs.input_ids[:, -self.max_context_length:]
            inputs.attention_mask = inputs.attention_mask[:, -self.max_context_length:]
        
        # Handle streaming generation
        if stream:
            return self._generate_stream(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                stop_strings=stop_strings,
                callback=callback,
            )
        
        # Standard generation
        with torch.no_grad():
            # Repeat inputs for multiple sequences if needed
            if num_return_sequences > 1:
                inputs.input_ids = inputs.input_ids.repeat(num_return_sequences, 1)
                inputs.attention_mask = inputs.attention_mask.repeat(num_return_sequences, 1)
            
            # Generate
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=inputs.input_ids.shape[1] + max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode the outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove the prompt from the generated texts
        prompt_length = len(prompt)
        generated_texts = [text[prompt_length:] for text in generated_texts]
        
        # Apply stop strings if provided
        if stop_strings:
            for i, text in enumerate(generated_texts):
                for stop_string in stop_strings:
                    if stop_string in text:
                        generated_texts[i] = text.split(stop_string)[0]
        
        # Return a single string if only one sequence was requested
        if num_return_sequences == 1:
            return generated_texts[0]
        
        return generated_texts
    
    def _generate_stream(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        stop_strings: Optional[List[str]] = None,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Iterator[str]:
        """Stream generation token by token.
        
        Args:
            inputs: Tokenized inputs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            stop_strings: List of strings to stop generation when encountered
            callback: Optional callback function
            
        Yields:
            Generated tokens one by one
        """
        # Initialize generation
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        past_key_values = None
        generated_tokens = []
        stop_generation = False
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):
            if stop_generation:
                break
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model.forward(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            # Get logits and past key values
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    for token_id in set(input_ids[i].tolist()):
                        if token_id != self.tokenizer.pad_token_id:
                            logits[i, token_id] /= repetition_penalty
            
            # Sample next token
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")
                
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Update input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((input_ids.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=-1)
            
            # Decode the new token
            new_token = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_tokens.append(new_token)
            
            # Yield the new token
            yield new_token
            
            # Call callback if provided
            if callback:
                callback(new_token)
            
            # Check for stop strings
            if stop_strings:
                # Join all generated tokens
                text = "".join(generated_tokens)
                for stop_string in stop_strings:
                    if stop_string in text:
                        stop_generation = True
                        break
            
            # Check for EOS token
            if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                break
    
    def complete_code(
        self,
        code_prompt: str,
        language: str = "python",
        max_new_tokens: int = 512,
        temperature: float = 0.2,  # Lower temperature for code
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Union[str, Iterator[str]]:
        """Complete code based on the prompt with streaming support.
        
        Args:
            code_prompt: Code prompt to complete
            language: Programming language
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            stream: Whether to stream the output token by token
            callback: Optional callback function for streaming mode
            
        Returns:
            Completed code or iterator of tokens
        """
        # Format prompt for code completion
        formatted_prompt = f"### Language: {language}\n### Code:\n{code_prompt}"
        
        # Generate with code-specific parameters
        stop_strings = ["###", "```"]
        
        return self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
            stop_strings=stop_strings,
            stream=stream,
            callback=callback,
        )
    
    def solve_math(
        self,
        problem: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        show_work: bool = True,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Union[str, Iterator[str]]:
        """Solve a mathematical problem with streaming support.
        
        Args:
            problem: Math problem to solve
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            show_work: Whether to show the work or just the answer
            stream: Whether to stream the output token by token
            callback: Optional callback function for streaming mode
            
        Returns:
            Solution to the problem or iterator of tokens
        """
        # Format prompt for math problem solving
        if show_work:
            formatted_prompt = f"Solve the following math problem, showing your work step-by-step:\n\n{problem}\n\nSolution:\n"
        else:
            formatted_prompt = f"Solve the following math problem:\n\n{problem}\n\nAnswer:\n"
        
        # Generate with math-specific parameters
        return self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.03,
            do_sample=True,
            num_return_sequences=1,
            stream=stream,
            callback=callback,
        )
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 512,
        temperature: float = 0.5,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
    ) -> Union[str, Iterator[str]]:
        """Translate text from source language to target language with streaming support.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., "en", "hi")
            target_lang: Target language code (e.g., "en", "hi")
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the output token by token
            callback: Optional callback function for streaming mode
            
        Returns:
            Translated text or iterator of tokens
        """
        # Format prompt for translation
        formatted_prompt = f"Translate the following {source_lang} text to {target_lang}:\n\n{text}\n\nTranslation:\n"
        
        # Generate with translation-specific parameters
        return self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            do_sample=True,
            num_return_sequences=1,
            stream=stream,
            callback=callback,
        )
    
    def batch_process(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> List[str]:
        """Process multiple prompts in a batch for higher throughput.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            List of generated texts
        """
        # Process in batches
        batch_size = min(len(prompts), self.batch_size)
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Tokenize prompts
            inputs = self.tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Truncate if needed
            if inputs.input_ids.shape[1] > self.max_context_length:
                logger.warning(f"Prompts are too long, truncating to {self.max_context_length} tokens")
                inputs.input_ids = inputs.input_ids[:, -self.max_context_length:]
                inputs.attention_mask = inputs.attention_mask[:, -self.max_context_length:]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=inputs.input_ids.shape[1] + max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode the outputs
            batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Remove the prompts from the generated texts
            for j, (prompt, result) in enumerate(zip(batch_prompts, batch_results)):
                prompt_length = len(prompt)
                batch_results[j] = result[prompt_length:]
            
            results.extend(batch_results)
        
        return results
