#!/usr/bin/env python3
"""
DeepShiva Inference Script

This script provides inference capabilities for the DeepShiva MoE model, including:
- Text generation with various sampling strategies
- Code completion
- Mathematical reasoning
- Multilingual support

It can be used as a standalone script or imported as a module.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.moe_model import DeepShivaMoE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DeepShivaInference:
    """Inference class for DeepShiva model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        precision: str = "fp16",
        max_context_length: int = 8192,
    ):
        """Initialize the inference engine.
        
        Args:
            model_path: Path to the model checkpoint
            device: Device to run inference on ("cuda" or "cpu")
            precision: Precision to use for inference ("fp32", "fp16", "int8", "int4")
            max_context_length: Maximum context length for the model
        """
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.max_context_length = max_context_length
        
        logger.info(f"Loading model from {model_path}")
        self._load_model()
        logger.info("Model loaded successfully")
    
    def _load_model(self):
        """Load the model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate precision
        if self.precision == "fp32":
            self.model = DeepShivaMoE.from_pretrained(self.model_path)
        elif self.precision == "fp16":
            self.model = DeepShivaMoE.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif self.precision == "int8":
            self.model = DeepShivaMoE.from_pretrained(
                self.model_path,
                load_in_8bit=True,
                device_map="auto",
            )
        elif self.precision == "int4":
            self.model = DeepShivaMoE.from_pretrained(
                self.model_path,
                load_in_4bit=True,
                device_map="auto",
            )
        else:
            raise ValueError(f"Unsupported precision: {self.precision}")
        
        # Move model to device if not using device_map
        if "device_map" not in self.precision:
            self.model.to(self.device)
        
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
    ) -> Union[str, List[str]]:
        """Generate text based on the prompt.
        
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
            
        Returns:
            Generated text or list of generated texts
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Truncate if needed
        if inputs.input_ids.shape[1] > self.max_context_length:
            logger.warning(f"Prompt is too long, truncating to {self.max_context_length} tokens")
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
                num_return_sequences=num_return_sequences,
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
    
    def complete_code(
        self,
        code_prompt: str,
        language: str = "python",
        max_new_tokens: int = 512,
        temperature: float = 0.2,  # Lower temperature for code
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
    ) -> str:
        """Complete code based on the prompt.
        
        Args:
            code_prompt: Code prompt to complete
            language: Programming language
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Completed code
        """
        # Format prompt for code completion
        formatted_prompt = f"### Language: {language}\n### Code:\n{code_prompt}"
        
        # Generate with code-specific parameters
        completed_code = self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
            stop_strings=["###", "```"],
        )
        
        return completed_code
    
    def solve_math(
        self,
        problem: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        show_work: bool = True,
    ) -> str:
        """Solve a mathematical problem.
        
        Args:
            problem: Math problem to solve
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            show_work: Whether to show the work or just the answer
            
        Returns:
            Solution to the problem
        """
        # Format prompt for math problem solving
        if show_work:
            formatted_prompt = f"Solve the following math problem, showing your work step-by-step:\n\n{problem}\n\nSolution:\n"
        else:
            formatted_prompt = f"Solve the following math problem:\n\n{problem}\n\nAnswer:\n"
        
        # Generate with math-specific parameters
        solution = self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.03,
            do_sample=True,
            num_return_sequences=1,
        )
        
        return solution
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_new_tokens: int = 512,
        temperature: float = 0.5,
    ) -> str:
        """Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., "en", "hi")
            target_lang: Target language code (e.g., "en", "hi")
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Translated text
        """
        # Format prompt for translation
        formatted_prompt = f"Translate the following {source_lang} text to {target_lang}:\n\n{text}\n\nTranslation:\n"
        
        # Generate with translation-specific parameters
        translation = self.generate(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            do_sample=True,
            num_return_sequences=1,
        )
        
        return translation


def parse_args():
    parser = argparse.ArgumentParser(description="DeepShiva Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for generation",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to input file containing prompts",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output file for generated text",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Penalty for repeating tokens",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling or greedy decoding",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8", "int4"],
        help="Precision to use for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "code", "math", "translate"],
        help="Inference mode",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="en",
        help="Source language for translation",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="hi",
        help="Target language for translation",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Programming language for code completion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize inference engine
    inference = DeepShivaInference(
        model_path=args.model_path,
        device=args.device,
        precision=args.precision,
    )
    
    # Get prompts from input file or command line
    prompts = []
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                prompts.append(line.strip())
    elif args.prompt:
        prompts.append(args.prompt)
    else:
        # Interactive mode
        print("Enter prompt (Ctrl+D to end):")
        try:
            while True:
                prompt = input("> ")
                prompts.append(prompt)
                
                # Process in interactive mode
                if args.mode == "generate":
                    output = inference.generate(
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=args.do_sample,
                    )
                elif args.mode == "code":
                    output = inference.complete_code(
                        code_prompt=prompt,
                        language=args.language,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                elif args.mode == "math":
                    output = inference.solve_math(
                        problem=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                elif args.mode == "translate":
                    output = inference.translate(
                        text=prompt,
                        source_lang=args.source_lang,
                        target_lang=args.target_lang,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                
                print("\nOutput:")
                print(output)
                print("\n" + "-"*50 + "\n")
        except EOFError:
            pass
        
        # Exit after interactive mode
        return
    
    # Process prompts in batch mode
    outputs = []
    for prompt in prompts:
        if args.mode == "generate":
            output = inference.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
            )
        elif args.mode == "code":
            output = inference.complete_code(
                code_prompt=prompt,
                language=args.language,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        elif args.mode == "math":
            output = inference.solve_math(
                problem=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        elif args.mode == "translate":
            output = inference.translate(
                text=prompt,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        
        outputs.append(output)
    
    # Write outputs to file or print to console
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for output in outputs:
                f.write(output + "\n")
    else:
        for i, output in enumerate(outputs):
            print(f"Prompt {i+1}:")
            print(prompts[i])
            print("\nOutput:")
            print(output)
            print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()
